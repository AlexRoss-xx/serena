import json
import logging
import os
import shutil
import subprocess
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, NotRequired, Self, TypedDict, Union, cast

from sensai.util.string import ToStringMixin

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language
from solidlsp.ls import ReferenceInSymbol as LSPReferenceInSymbol
from solidlsp.ls_exceptions import SolidLSPException
from solidlsp.ls_types import Position, SymbolKind, UnifiedSymbolInformation

from .ls_manager import LanguageServerManager
from .project import Project

if TYPE_CHECKING:
    from .agent import SerenaAgent

log = logging.getLogger(__name__)
NAME_PATH_SEP = "/"


@dataclass
class LanguageServerSymbolLocation:
    """
    Represents the (start) location of a symbol identifier, which, within Serena, uniquely identifies the symbol.
    """

    relative_path: str | None
    """
    the relative path of the file containing the symbol; if None, the symbol is defined outside of the project's scope
    """
    line: int | None
    """
    the line number in which the symbol identifier is defined (if the symbol is a function, class, etc.);
    may be None for some types of symbols (e.g. SymbolKind.File)
    """
    column: int | None
    """
    the column number in which the symbol identifier is defined (if the symbol is a function, class, etc.);
    may be None for some types of symbols (e.g. SymbolKind.File)
    """

    def __post_init__(self) -> None:
        if self.relative_path is not None:
            self.relative_path = self.relative_path.replace("/", os.path.sep)

    def to_dict(self, include_relative_path: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if not include_relative_path:
            result.pop("relative_path", None)
        return result

    def has_position_in_file(self) -> bool:
        return self.relative_path is not None and self.line is not None and self.column is not None


@dataclass
class PositionInFile:
    """
    Represents a character position within a file
    """

    line: int
    """
    the 0-based line number in the file
    """
    col: int
    """
    the 0-based column
    """

    def to_lsp_position(self) -> Position:
        """
        Convert to LSP Position.
        """
        return Position(line=self.line, character=self.col)


class Symbol(ToStringMixin, ABC):
    @abstractmethod
    def get_body_start_position(self) -> PositionInFile | None:
        pass

    @abstractmethod
    def get_body_end_position(self) -> PositionInFile | None:
        pass

    def get_body_start_position_or_raise(self) -> PositionInFile:
        """
        Get the start position of the symbol body, raising an error if it is not defined.
        """
        pos = self.get_body_start_position()
        if pos is None:
            raise ValueError(f"Body start position is not defined for {self}")
        return pos

    def get_body_end_position_or_raise(self) -> PositionInFile:
        """
        Get the end position of the symbol body, raising an error if it is not defined.
        """
        pos = self.get_body_end_position()
        if pos is None:
            raise ValueError(f"Body end position is not defined for {self}")
        return pos

    @abstractmethod
    def is_neighbouring_definition_separated_by_empty_line(self) -> bool:
        """
        :return: whether a symbol definition of this symbol's kind is usually separated from the
            previous/next definition by at least one empty line.
        """


class NamePathMatcher(ToStringMixin):
    """
    Matches name paths of symbols against search patterns.

    A name path is a path in the symbol tree *within a source file*.
    For example, the method `my_method` defined in class `MyClass` would have the name path `MyClass/my_method`.
    If a symbol is overloaded (e.g., in Java), a 0-based index is appended (e.g. "MyClass/my_method[0]") to
    uniquely identify it.

    A matching pattern can be:
     * a simple name (e.g. "method"), which will match any symbol with that name
     * a relative path like "class/method", which will match any symbol with that name path suffix
     * an absolute name path "/class/method" (absolute name path), which requires an exact match of the full name path within the source file.
    Append an index `[i]` to match a specific overload only, e.g. "MyClass/my_method[1]".
    """

    def __init__(self, name_path_pattern: str, substring_matching: bool) -> None:
        """
        :param name_path_pattern: the name path expression to match against
        :param substring_matching: whether to use substring matching for the last segment
        """
        assert name_path_pattern, "name_path must not be empty"
        self._expr = name_path_pattern
        self._substring_matching = substring_matching
        self._is_absolute_pattern = name_path_pattern.startswith(NAME_PATH_SEP)
        self._pattern_parts = name_path_pattern.lstrip(NAME_PATH_SEP).rstrip(NAME_PATH_SEP).split(NAME_PATH_SEP)

        # extract overload index "[idx]" if present at end of last part
        self._overload_idx: int | None = None
        last_part = self._pattern_parts[-1]
        if last_part.endswith("]") and "[" in last_part:
            bracket_idx = last_part.rfind("[")
            index_part = last_part[bracket_idx + 1 : -1]
            if index_part.isdigit():
                self._pattern_parts[-1] = last_part[:bracket_idx]
                self._overload_idx = int(index_part)

    def _tostring_includes(self) -> list[str]:
        return ["_expr"]

    def matches_ls_symbol(self, symbol: "LanguageServerSymbol") -> bool:
        return self.matches_components(symbol.get_name_path_parts(), symbol.overload_idx)

    def matches_components(self, symbol_name_path_parts: list[str], overload_idx: int | None) -> bool:
        # filtering based on ancestors
        if len(self._pattern_parts) > len(symbol_name_path_parts):
            # can't possibly match if pattern has more parts than symbol
            return False
        if self._is_absolute_pattern and len(self._pattern_parts) != len(symbol_name_path_parts):
            # for absolute patterns, the number of parts must match exactly
            return False
        if symbol_name_path_parts[-len(self._pattern_parts) : -1] != self._pattern_parts[:-1]:
            # ancestors must match
            return False

        # matching the last part of the symbol name
        name_to_match = self._pattern_parts[-1]
        symbol_name = symbol_name_path_parts[-1]
        if self._substring_matching:
            if name_to_match not in symbol_name:
                return False
        else:
            if name_to_match != symbol_name:
                return False

        # check for matching overload index
        if self._overload_idx is not None:
            if overload_idx != self._overload_idx:
                return False

        return True


class LanguageServerSymbol(Symbol, ToStringMixin):
    def __init__(self, symbol_root_from_ls: UnifiedSymbolInformation) -> None:
        self.symbol_root = symbol_root_from_ls

    def _tostring_includes(self) -> list[str]:
        return []

    def _tostring_additional_entries(self) -> dict[str, Any]:
        return dict(name=self.name, kind=self.kind, num_children=len(self.symbol_root["children"]))

    @property
    def name(self) -> str:
        return self.symbol_root["name"]

    @property
    def kind(self) -> str:
        return SymbolKind(self.symbol_kind).name

    @property
    def symbol_kind(self) -> SymbolKind:
        return self.symbol_root["kind"]

    def is_low_level(self) -> bool:
        """
        :return: whether the symbol is a low-level symbol (variable, constant, etc.), which typically represents data
            rather than structure and therefore is not relevant in a high-level overview of the code.
        """
        return self.symbol_kind >= SymbolKind.Variable.value

    @property
    def overload_idx(self) -> int | None:
        return self.symbol_root.get("overload_idx")

    def is_neighbouring_definition_separated_by_empty_line(self) -> bool:
        return self.symbol_kind in (SymbolKind.Function, SymbolKind.Method, SymbolKind.Class, SymbolKind.Interface, SymbolKind.Struct)

    @property
    def relative_path(self) -> str | None:
        location = self.symbol_root.get("location")
        if location:
            return location.get("relativePath")
        return None

    @property
    def location(self) -> LanguageServerSymbolLocation:
        """
        :return: the start location of the actual symbol identifier
        """
        return LanguageServerSymbolLocation(relative_path=self.relative_path, line=self.line, column=self.column)

    @property
    def body_start_position(self) -> Position | None:
        location = self.symbol_root.get("location")
        if location:
            range_info = location.get("range")
            if range_info:
                start_pos = range_info.get("start")
                if start_pos:
                    return start_pos
        return None

    @property
    def body_end_position(self) -> Position | None:
        location = self.symbol_root.get("location")
        if location:
            range_info = location.get("range")
            if range_info:
                end_pos = range_info.get("end")
                if end_pos:
                    return end_pos
        return None

    def get_body_start_position(self) -> PositionInFile | None:
        start_pos = self.body_start_position
        if start_pos is None:
            return None
        return PositionInFile(line=start_pos["line"], col=start_pos["character"])

    def get_body_end_position(self) -> PositionInFile | None:
        end_pos = self.body_end_position
        if end_pos is None:
            return None
        return PositionInFile(line=end_pos["line"], col=end_pos["character"])

    def get_body_line_numbers(self) -> tuple[int | None, int | None]:
        start_pos = self.body_start_position
        end_pos = self.body_end_position
        start_line = start_pos["line"] if start_pos else None
        end_line = end_pos["line"] if end_pos else None
        return start_line, end_line

    @property
    def line(self) -> int | None:
        """
        :return: the line in which the symbol identifier is defined.
        """
        if "selectionRange" in self.symbol_root:
            return self.symbol_root["selectionRange"]["start"]["line"]
        else:
            # line is expected to be undefined for some types of symbols (e.g. SymbolKind.File)
            return None

    @property
    def column(self) -> int | None:
        if "selectionRange" in self.symbol_root:
            return self.symbol_root["selectionRange"]["start"]["character"]
        else:
            # precise location is expected to be undefined for some types of symbols (e.g. SymbolKind.File)
            return None

    @property
    def body(self) -> str | None:
        return self.symbol_root.get("body")

    def get_name_path(self) -> str:
        """
        Get the name path of the symbol, e.g. "class/method/inner_function" or
        "class/method[1]" (overloaded method with identifying index).
        """
        name_path = NAME_PATH_SEP.join(self.get_name_path_parts())
        if "overload_idx" in self.symbol_root:
            name_path += f"[{self.symbol_root['overload_idx']}]"
        return name_path

    def get_name_path_parts(self) -> list[str]:
        """
        Get the parts of the name path of the symbol (e.g. ["class", "method", "inner_function"]).
        """
        ancestors_within_file = list(self.iter_ancestors(up_to_symbol_kind=SymbolKind.File))
        ancestors_within_file.reverse()
        return [a.name for a in ancestors_within_file] + [self.name]

    def iter_children(self) -> Iterator[Self]:
        for c in self.symbol_root["children"]:
            yield self.__class__(c)

    def iter_ancestors(self, up_to_symbol_kind: SymbolKind | None = None) -> Iterator[Self]:
        """
        Iterate over all ancestors of the symbol, starting with the parent and going up to the root or
        the given symbol kind.

        :param up_to_symbol_kind: if provided, iteration will stop *before* the first ancestor of the given kind.
            A typical use case is to pass `SymbolKind.File` or `SymbolKind.Package`.
        """
        parent = self.get_parent()
        if parent is not None:
            if up_to_symbol_kind is None or parent.symbol_kind != up_to_symbol_kind:
                yield parent
                yield from parent.iter_ancestors(up_to_symbol_kind=up_to_symbol_kind)

    def get_parent(self) -> Self | None:
        parent_root = self.symbol_root.get("parent")
        if parent_root is None:
            return None
        return self.__class__(parent_root)

    def find(
        self,
        name_path_pattern: str,
        substring_matching: bool = False,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
    ) -> list[Self]:
        """
        Find all symbols within the symbol's subtree that match the given name path pattern.

        :param name_path_pattern: the name path pattern to match against (see class :class:`NamePathMatcher` for details)
        :param substring_matching: whether to use substring matching (as opposed to exact matching)
            of the last segment of `name_path` against the symbol name.
        :param include_kinds: an optional sequence of ints representing the LSP symbol kind.
            If provided, only symbols of the given kinds will be included in the result.
        :param exclude_kinds: If provided, symbols of the given kinds will be excluded from the result.
        """
        result = []
        name_path_matcher = NamePathMatcher(name_path_pattern, substring_matching)

        def should_include(s: "LanguageServerSymbol") -> bool:
            if include_kinds is not None and s.symbol_kind not in include_kinds:
                return False
            if exclude_kinds is not None and s.symbol_kind in exclude_kinds:
                return False
            return name_path_matcher.matches_ls_symbol(s)

        def traverse(s: "LanguageServerSymbol") -> None:
            if should_include(s):
                result.append(s)
            for c in s.iter_children():
                traverse(c)

        traverse(self)
        return result

    def to_dict(
        self,
        kind: bool = False,
        location: bool = False,
        depth: int = 0,
        include_body: bool = False,
        include_children_body: bool = False,
        include_relative_path: bool = True,
        child_inclusion_predicate: Callable[[Self], bool] | None = None,
    ) -> dict[str, Any]:
        """
        Converts the symbol to a dictionary.

        :param kind: whether to include the kind of the symbol
        :param location: whether to include the location of the symbol
        :param depth: the depth up to which to include child symbols (0 = do not include children)
        :param include_body: whether to include the body of the top-level symbol.
        :param include_children_body: whether to also include the body of the children.
            Note that the body of the children is part of the body of the parent symbol,
            so there is usually no need to set this to True unless you want process the output
            and pass the children without passing the parent body to the LM.
        :param include_relative_path: whether to include the relative path of the symbol in the location
            entry. Relative paths of the symbol's children are always excluded.
        :param child_inclusion_predicate: an optional predicate that decides whether a child symbol
            should be included.
        :return: a dictionary representation of the symbol
        """
        result: dict[str, Any] = {"name": self.name, "name_path": self.get_name_path()}

        if kind:
            result["kind"] = self.kind

        if location:
            result["location"] = self.location.to_dict(include_relative_path=include_relative_path)
            body_start_line, body_end_line = self.get_body_line_numbers()
            result["body_location"] = {"start_line": body_start_line, "end_line": body_end_line}

        if include_body:
            if self.body is None:
                log.warning("Requested body for symbol, but it is not present. The symbol might have been loaded with include_body=False.")
            result["body"] = self.body

        if child_inclusion_predicate is None:
            child_inclusion_predicate = lambda s: True

        def included_children(s: Self) -> list[dict[str, Any]]:
            children = []
            for c in s.iter_children():
                if not child_inclusion_predicate(c):
                    continue
                children.append(
                    c.to_dict(
                        kind=kind,
                        location=location,
                        depth=depth - 1,
                        child_inclusion_predicate=child_inclusion_predicate,
                        include_body=include_children_body,
                        include_children_body=include_children_body,
                        # all children have the same relative path as the parent
                        include_relative_path=False,
                    )
                )
            return children

        if depth > 0:
            children = included_children(self)
            if len(children) > 0:
                result["children"] = included_children(self)

        return result


@dataclass
class ReferenceInLanguageServerSymbol(ToStringMixin):
    """
    Represents the location of a reference to another symbol within a symbol/file.

    The contained symbol is the symbol within which the reference is located,
    not the symbol that is referenced.
    """

    symbol: LanguageServerSymbol
    """
    the symbol within which the reference is located
    """
    line: int
    """
    the line number in which the reference is located (0-based)
    """
    character: int
    """
    the column number in which the reference is located (0-based)
    """

    @classmethod
    def from_lsp_reference(cls, reference: LSPReferenceInSymbol) -> Self:
        return cls(symbol=LanguageServerSymbol(reference.symbol), line=reference.line, character=reference.character)

    def get_relative_path(self) -> str | None:
        return self.symbol.location.relative_path


class LanguageServerSymbolRetriever:
    def __init__(self, ls: SolidLanguageServer | LanguageServerManager, agent: Union["SerenaAgent", None] = None) -> None:
        """
        :param ls: the language server or language server manager to use for symbol retrieval and editing operations.
        :param agent: the agent to use (only needed for marking files as modified). You can pass None if you don't
            need an agent to be aware of file modifications performed by the symbol manager.
        """
        if isinstance(ls, SolidLanguageServer):
            ls_manager = LanguageServerManager({ls.language: ls})
        else:
            ls_manager = ls
        assert isinstance(ls_manager, LanguageServerManager)
        self._ls_manager: LanguageServerManager = ls_manager
        self.agent = agent

    @staticmethod
    def _is_simple_name_pattern(name_path_pattern: str) -> bool:
        """
        :return: True if the pattern is a single-segment name (optionally absolute), without overload selector.
            Examples: "Foo", "/Foo"
            Non-examples: "Foo/bar", "Foo[0]", "/Foo/bar"
        """
        expr = name_path_pattern.strip()
        if not expr:
            return False
        # IMPORTANT: patterns starting with "/" are absolute name paths within a file.
        # WorkspaceSymbol results usually don't include reliable parent/container hierarchy,
        # so we avoid the WorkspaceSymbol fast path for absolute patterns to preserve correctness.
        if expr.startswith(NAME_PATH_SEP):
            return False
        expr = expr.lstrip(NAME_PATH_SEP).rstrip(NAME_PATH_SEP)
        if not expr:
            return False
        if NAME_PATH_SEP in expr:
            return False
        # Reject overload selectors like "Foo[0]"
        if expr.endswith("]") and "[" in expr:
            bracket_idx = expr.rfind("[")
            index_part = expr[bracket_idx + 1 : -1]
            if index_part.isdigit():
                return False
        return True

    def get_root_path(self) -> str:
        return self._ls_manager.get_root_path()

    def get_language_server(self, relative_path: str) -> SolidLanguageServer:
        return self._ls_manager.get_language_server(relative_path)

    def _call_with_restart_on_terminated(
        self, lang_server: SolidLanguageServer, fn: Callable[[SolidLanguageServer], Any]
    ) -> Any:
        """
        If a request fails because the LS process terminated, restart that LS and retry once.
        This makes the system resilient against native crashes (e.g. pasls "Access violation").
        """
        try:
            return fn(lang_server)
        except SolidLSPException as e:
            if not e.is_language_server_terminated():
                raise
            # Restart the affected language server and retry once
            restarted = self._ls_manager.restart_language_server(lang_server.language)
            return fn(restarted)

    def find(
        self,
        name_path_pattern: str,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
        substring_matching: bool = False,
        within_relative_path: str | None = None,
    ) -> list[LanguageServerSymbol]:
        """
        Finds all symbols that match the given name path pattern (see class :class:`NamePathMatcher` for details),
        optionally limited to a specific file and filtered by kind.
        """
        symbols: list[LanguageServerSymbol] = []

        # Fast path: if the query is just a single name, prefer LSP workspace/symbol.
        # This avoids walking all files (request_full_symbol_tree) and is typically much faster on large projects.
        if self._is_simple_name_pattern(name_path_pattern):
            query = name_path_pattern.strip().lstrip(NAME_PATH_SEP).rstrip(NAME_PATH_SEP)
            for lang_server in self._ls_manager.iter_language_servers():
                try:
                    ws = lang_server.request_workspace_symbol(query=query)
                except Exception:
                    # Some servers throw if the workspace isn't configured (e.g. TS "No Project").
                    # Treat this as "unsupported" and fall back to other strategies.
                    continue
                if not ws:
                    continue

                for s_dict in ws:
                    # optional scope restriction: filter by path prefix (directory) or exact file (file)
                    if within_relative_path:
                        loc: dict[str, Any] = cast(dict[str, Any], s_dict.get("location") or {})
                        rel = loc.get("relativePath")
                        if isinstance(rel, str):
                            # normalize separators for prefix comparisons
                            rel_norm = rel.replace("\\", "/")
                            scope_norm = within_relative_path.replace("\\", "/").rstrip("/")
                            if scope_norm and not (rel_norm == scope_norm or rel_norm.startswith(scope_norm + "/")):
                                continue

                    # kind filters (workspace symbols use numeric kinds)
                    kind_val = s_dict.get("kind")
                    if include_kinds is not None and kind_val not in include_kinds:
                        continue
                    if exclude_kinds is not None and kind_val in exclude_kinds:
                        continue

                    ls_symbol = LanguageServerSymbol(s_dict)
                    # apply our name-path matcher semantics (exact vs substring)
                    if ls_symbol.find(name_path_pattern, substring_matching=substring_matching):
                        symbols.append(ls_symbol)

            if symbols:
                return symbols

            # Prefilter fallback: if workspace/symbol yields nothing (unsupported/disabled), avoid a full workspace scan.
            # We try to narrow candidate files using ripgrep (if available), then only request document symbols for those files.
            # This provides a major speedup on large workspaces where only a handful of files contain the identifier.
            candidates = self._prefilter_candidate_files(query=query, within_relative_path=within_relative_path)
            if candidates:
                for lang_server in self._ls_manager.iter_language_servers():
                    for rel_file in candidates:
                        try:
                            if lang_server.is_ignored_path(rel_file):
                                continue
                        except Exception:
                            # best-effort filtering
                            pass
                        try:
                            symbol_roots = self._call_with_restart_on_terminated(
                                lang_server, lambda ls: ls.request_full_symbol_tree(within_relative_path=rel_file)
                            )
                        except Exception:
                            continue
                        for root in symbol_roots:
                            symbols.extend(
                                LanguageServerSymbol(root).find(
                                    name_path_pattern,
                                    include_kinds=include_kinds,
                                    exclude_kinds=exclude_kinds,
                                    substring_matching=substring_matching,
                                )
                            )
                if symbols:
                    return symbols

            # Robustness guard (Pascal): if we couldn't narrow candidates cheaply, avoid the full workspace scan.
            # Full scans are both extremely slow on large Pascal workspaces and can crash unstable pasls builds.
            try:
                if any(ls.language == Language.PASCAL for ls in self._ls_manager.iter_language_servers()):
                    return []
            except Exception:
                # If we can't determine language, fall through to generic behavior.
                pass

        # Fallback: full scan via symbol tree
        for lang_server in self._ls_manager.iter_language_servers():
            try:
                symbol_roots = self._call_with_restart_on_terminated(
                    lang_server, lambda ls: ls.request_full_symbol_tree(within_relative_path=within_relative_path)
                )
            except Exception:
                continue
            for root in symbol_roots:
                symbols.extend(
                    LanguageServerSymbol(root).find(
                        name_path_pattern,
                        include_kinds=include_kinds,
                        exclude_kinds=exclude_kinds,
                        substring_matching=substring_matching,
                    )
                )
        return symbols

    def _prefilter_candidate_files(self, *, query: str, within_relative_path: str | None) -> list[str]:
        """
        Best-effort candidate file prefilter for cases where workspace/symbol is unavailable.

        Uses ripgrep if present; otherwise falls back to `git grep` (fast on large repos), and finally
        to a bounded filesystem scan.

        Returns relative paths (OS-native separators ok).
        """
        query = (query or "").strip()
        if not query:
            return []

        root = self.get_root_path()
        scope_rel = within_relative_path or "."
        scope_abs = os.path.join(root, scope_rel) if within_relative_path else root

        # If caller already restricts to a file, we're done.
        if within_relative_path and os.path.isfile(scope_abs):
            return [within_relative_path]

        # Keep the candidate set bounded to avoid pathological cases.
        # NOTE: This is a *prefilter* only. Keeping this number small is important to avoid accidentally
        # dragging in thousands of files and turning this into "full scan" behavior.
        max_candidates = 50

        # Limit to likely source files for symbol definitions. For Pascal/Delphi workspaces, `.dfm` files are
        # extremely common and often contain many incidental string matches; including them can explode runtime.
        # (If a symbol truly only exists inside a .dfm, it's not a code symbol Serena can edit anyway.)
        include_globs = [
            "*.pas",
            "*.pp",
            "*.inc",
        ]

        def _normalize_and_cap(lines: list[str]) -> list[str]:
            paths: list[str] = []
            for line in lines:
                p = (line or "").strip()
                if not p:
                    continue
                # Normalize to a relative path under root if a tool returned absolute paths
                if os.path.isabs(p):
                    try:
                        p = os.path.relpath(p, root)
                    except Exception:
                        continue
                paths.append(p)
                if len(paths) >= max_candidates:
                    break
            return paths

        def _is_allowed_source_path(rel_path: str) -> bool:
            # Keep in sync with include_globs above (extensions only).
            _, ext = os.path.splitext(rel_path.lower())
            return ext in {".pas", ".pp", ".inc"}

        # 1) Prefer ripgrep (fastest, supports filesystem changes).
        rg = shutil.which("rg")
        if rg:
            try:
                proc = subprocess.run(
                    [
                        rg,
                        "--files-with-matches",
                        "--fixed-strings",
                        "--no-messages",
                        "--hidden",
                        "--follow",
                        *[g for glob in include_globs for g in ("--glob", glob)],
                        query,
                        scope_rel,
                    ],
                    cwd=root,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except Exception:
                proc = None

            if proc and proc.returncode in (0, 1):  # 0=matches, 1=no matches
                candidates = _normalize_and_cap((proc.stdout or "").splitlines())
                if candidates:
                    return candidates
                # Pascal is case-insensitive; try ignore-case if a case-sensitive search didn't match.
                try:
                    proc_i = subprocess.run(
                        [
                            rg,
                            "--files-with-matches",
                            "--fixed-strings",
                            "--no-messages",
                            "--hidden",
                            "--follow",
                            "--ignore-case",
                            *[g for glob in include_globs for g in ("--glob", glob)],
                            query,
                            scope_rel,
                        ],
                        cwd=root,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if proc_i.returncode in (0, 1):
                        candidates_i = _normalize_and_cap((proc_i.stdout or "").splitlines())
                        if candidates_i:
                            return candidates_i
                except Exception:
                    pass

        # 2) Fall back to git grep (very fast on huge repos; searches tracked files).
        git = shutil.which("git")
        if git:
            try:
                # NOTE: We filter extensions in Python to keep the git invocation simple and robust across platforms.
                proc = subprocess.run(
                    [
                        git,
                        "-C",
                        root,
                        "grep",
                        "-l",
                        "-I",
                        "--fixed-strings",
                        "--no-color",
                        "--no-messages",
                        "--",
                        query,
                        scope_rel,
                    ],
                    cwd=root,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except Exception:
                proc = None

            if proc and proc.returncode in (0, 1):
                lines = [ln for ln in (proc.stdout or "").splitlines() if _is_allowed_source_path(ln)]
                candidates = _normalize_and_cap(lines)
                if candidates:
                    return candidates

                # Pascal is case-insensitive; try ignore-case if needed.
                try:
                    proc_i = subprocess.run(
                        [
                            git,
                            "-C",
                            root,
                            "grep",
                            "-l",
                            "-I",
                            "-i",
                            "--fixed-strings",
                            "--no-color",
                            "--no-messages",
                            "--",
                            query,
                            scope_rel,
                        ],
                        cwd=root,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if proc_i.returncode in (0, 1):
                        lines_i = [ln for ln in (proc_i.stdout or "").splitlines() if _is_allowed_source_path(ln)]
                        candidates_i = _normalize_and_cap(lines_i)
                        if candidates_i:
                            return candidates_i
                except Exception:
                    pass

        # 3) Last resort: bounded filesystem scan (covers untracked files / repos without git).
        # Keep this conservative to avoid turning into a full scan.
        try:
            root_abs = root
            start_dir = scope_abs if os.path.isdir(scope_abs) else root_abs
            scanned = 0
            found: list[str] = []
            for dirpath, dirnames, filenames in os.walk(start_dir):
                # Prune heavy/irrelevant directories.
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in {".git", ".serena", "__pycache__", ".venv", "node_modules"}
                    and not d.startswith(".mypy_cache")
                    and not d.startswith(".pytest_cache")
                ]
                for fn in filenames:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in {".pas", ".pp", ".inc"}:
                        continue
                    scanned += 1
                    if scanned > 2000:
                        break
                    abs_p = os.path.join(dirpath, fn)
                    try:
                        with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read(256_000)  # cap per-file read; enough for unit header + typical symbol refs
                    except Exception:
                        continue
                    if query in content or query.lower() in content.lower():
                        try:
                            rel = os.path.relpath(abs_p, root_abs)
                        except Exception:
                            continue
                        found.append(rel)
                        if len(found) >= max_candidates:
                            break
                if scanned > 2000 or len(found) >= max_candidates:
                    break
            return _normalize_and_cap(found)
        except Exception:
            return []

    def find_unique(
        self,
        name_path_pattern: str,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
        substring_matching: bool = False,
        within_relative_path: str | None = None,
    ) -> LanguageServerSymbol:
        # For single-file scopes, prefer document symbols for correctness.
        # WorkspaceSymbol results often have identifier-only ranges which break edit operations.
        if within_relative_path:
            try:
                abs_scope = os.path.join(self.get_root_path(), within_relative_path)
                if os.path.isfile(abs_scope):
                    symbol_candidates: list[LanguageServerSymbol] = []
                    lang_server = self.get_language_server(within_relative_path)
                    for root in lang_server.request_full_symbol_tree(within_relative_path=within_relative_path):
                        symbol_candidates.extend(
                            LanguageServerSymbol(root).find(
                                name_path_pattern,
                                include_kinds=include_kinds,
                                exclude_kinds=exclude_kinds,
                                substring_matching=substring_matching,
                            )
                        )
                else:
                    symbol_candidates = self.find(
                        name_path_pattern,
                        include_kinds=include_kinds,
                        exclude_kinds=exclude_kinds,
                        substring_matching=substring_matching,
                        within_relative_path=within_relative_path,
                    )
            except Exception:
                symbol_candidates = self.find(
                    name_path_pattern,
                    include_kinds=include_kinds,
                    exclude_kinds=exclude_kinds,
                    substring_matching=substring_matching,
                    within_relative_path=within_relative_path,
                )
        else:
            symbol_candidates = self.find(
                name_path_pattern,
                include_kinds=include_kinds,
                exclude_kinds=exclude_kinds,
                substring_matching=substring_matching,
                within_relative_path=within_relative_path,
            )
        if len(symbol_candidates) == 1:
            return symbol_candidates[0]
        elif len(symbol_candidates) == 0:
            raise ValueError(f"No symbol matching '{name_path_pattern}' found")
        else:
            # There are multiple candidates.
            # If only one of the candidates has the given pattern as its exact name path, return that one
            exact_matches = [s for s in symbol_candidates if s.get_name_path() == name_path_pattern]
            if len(exact_matches) == 1:
                return exact_matches[0]
            # otherwise, raise an error
            include_rel_path = within_relative_path is not None
            raise ValueError(
                f"Found multiple {len(symbol_candidates)} symbols matching '{name_path_pattern}'. "
                "They are: \n"
                + json.dumps([s.to_dict(kind=True, include_relative_path=include_rel_path) for s in symbol_candidates], indent=2)
            )

    def find_by_location(self, location: LanguageServerSymbolLocation) -> LanguageServerSymbol | None:
        rel_path = location.relative_path
        if rel_path is None:
            return None
        lang_server = self.get_language_server(rel_path)
        document_symbols = self._call_with_restart_on_terminated(lang_server, lambda ls: ls.request_document_symbols(rel_path))
        for symbol_dict in document_symbols.iter_symbols():
            symbol = LanguageServerSymbol(symbol_dict)
            if symbol.location == location:
                return symbol
        return None

    def find_referencing_symbols(
        self,
        name_path: str,
        relative_file_path: str,
        include_body: bool = False,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
    ) -> list[ReferenceInLanguageServerSymbol]:
        """
        Find all symbols that reference the specified symbol, which is assumed to be unique.

        :param name_path: the name path of the symbol to find. (While this can be a matching pattern, it should
            usually be the full path to ensure uniqueness.)
        :param relative_file_path: the relative path of the file in which the referenced symbol is defined.
        :param include_body: whether to include the body of all symbols in the result.
            Not recommended, as the referencing symbols will often be files, and thus the bodies will be very long.
        :param include_kinds: which kinds of symbols to include in the result.
        :param exclude_kinds: which kinds of symbols to exclude from the result.
        """
        symbol = self.find_unique(name_path, substring_matching=False, within_relative_path=relative_file_path)
        return self.find_referencing_symbols_by_location(
            symbol.location,
            referenced_symbol_name=symbol.name,
            include_body=include_body,
            include_kinds=include_kinds,
            exclude_kinds=exclude_kinds,
        )

    def find_referencing_symbols_by_location(
        self,
        symbol_location: LanguageServerSymbolLocation,
        referenced_symbol_name: str | None = None,
        include_body: bool = False,
        include_kinds: Sequence[SymbolKind] | None = None,
        exclude_kinds: Sequence[SymbolKind] | None = None,
    ) -> list[ReferenceInLanguageServerSymbol]:
        """
        Find all symbols that reference the symbol at the given location.

        :param symbol_location: the location of the symbol for which to find references.
            Does not need to include an end_line, as it is unused in the search.
        :param include_body: whether to include the body of all symbols in the result.
            Not recommended, as the referencing symbols will often be files, and thus the bodies will be very long.
            Note: you can filter out the bodies of the children if you set include_children_body=False
            in the to_dict method.
        :param include_kinds: an optional sequence of ints representing the LSP symbol kind.
            If provided, only symbols of the given kinds will be included in the result.
        :param exclude_kinds: If provided, symbols of the given kinds will be excluded from the result.
            Takes precedence over include_kinds.
        :return: a list of symbols that reference the given symbol
        """
        if not symbol_location.has_position_in_file():
            raise ValueError("Symbol location does not contain a valid position in a file")
        assert symbol_location.relative_path is not None
        assert symbol_location.line is not None
        assert symbol_location.column is not None
        lang_server = self.get_language_server(symbol_location.relative_path)
        try:
            references = lang_server.request_referencing_symbols(
                relative_file_path=symbol_location.relative_path,
                line=symbol_location.line,
                column=symbol_location.column,
                include_imports=False,
                include_self=False,
                include_body=include_body,
                include_file_symbols=True,
            )
        except Exception as e:
            # pasls can crash (native "Access violation") on textDocument/references for some symbols/projects.
            # Provide a robust, fast fallback: text search for the identifier in Pascal source files.
            if getattr(lang_server, "language", None) == Language.PASCAL:
                query = (referenced_symbol_name or "").strip()
                if not query:
                    return []
                return self._fallback_references_text_search(
                    query=query,
                    include_kinds=include_kinds,
                    exclude_kinds=exclude_kinds,
                )
            raise

        if include_kinds is not None:
            references = [s for s in references if s.symbol["kind"] in include_kinds]

        if exclude_kinds is not None:
            references = [s for s in references if s.symbol["kind"] not in exclude_kinds]

        return [ReferenceInLanguageServerSymbol.from_lsp_reference(r) for r in references]

    def _fallback_references_text_search(
        self,
        *,
        query: str,
        include_kinds: Sequence[SymbolKind] | None,
        exclude_kinds: Sequence[SymbolKind] | None,
    ) -> list[ReferenceInLanguageServerSymbol]:
        """
        Fallback for unstable Pascal LS builds: approximate references using text search.

        Returns file-level "symbols" with reference coordinates. This is meant to be robust and fast, not perfect.
        """
        # Use the same candidate prefilter machinery (rg/git grep + extension filtering).
        candidates = self._prefilter_candidate_files(query=query, within_relative_path=None)
        if not candidates:
            return []

        # Bound work to keep this predictable.
        max_files = 25
        max_total_refs = 200
        max_refs_per_file = 20

        root = self.get_root_path()

        # Prefer identifier-like matches (word boundary). Pascal identifiers are [A-Za-z_][A-Za-z0-9_]*.
        # We still fall back to substring match if the identifier boundary regex doesn't match.
        ident_re = re.compile(rf"(?i)(?<![A-Za-z0-9_]){re.escape(query)}(?![A-Za-z0-9_])")

        results: list[ReferenceInLanguageServerSymbol] = []
        for rel_file in candidates[:max_files]:
            abs_file = os.path.join(root, rel_file)
            try:
                with open(abs_file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.read().splitlines()
            except Exception:
                continue

            refs_in_file = 0
            for i, line in enumerate(lines):
                if refs_in_file >= max_refs_per_file or len(results) >= max_total_refs:
                    break
                m = ident_re.search(line)
                if not m:
                    # Last resort: cheap substring (keeps behavior close to previous "code search" expectations)
                    idx = line.lower().find(query.lower())
                    if idx < 0:
                        continue
                    col = idx
                else:
                    col = m.start()

                file_symbol: UnifiedSymbolInformation = {
                    "name": os.path.basename(rel_file),
                    "kind": SymbolKind.File,
                    "location": {
                        "relativePath": rel_file,
                        "absolutePath": abs_file,
                        "uri": "",
                        "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}},
                    },
                    "children": [],
                }
                results.append(ReferenceInLanguageServerSymbol(symbol=LanguageServerSymbol(file_symbol), line=i, character=col))
                refs_in_file += 1

            if len(results) >= max_total_refs:
                break

        # include/exclude kinds apply to the *referencing symbol*; here it's always a File kind.
        if include_kinds is not None and SymbolKind.File not in include_kinds:
            return []
        if exclude_kinds is not None and SymbolKind.File in exclude_kinds:
            return []
        return results

    def get_symbol_overview(self, relative_path: str, depth: int = 0) -> dict[str, list[dict]]:
        """
        :param relative_path: the path of the file or directory for which to get the symbol overview
        :param depth: the depth up to which to include child symbols (0 = only top-level symbols)
        :return: a mapping from file paths to lists of symbol dictionaries.
            For the case where a file is passed, the mapping will contain a single entry.
        """
        lang_server = self.get_language_server(relative_path)
        path_to_unified_symbols = self._call_with_restart_on_terminated(lang_server, lambda ls: ls.request_overview(relative_path))

        def child_inclusion_predicate(s: LanguageServerSymbol) -> bool:
            return not s.is_low_level()

        result = {}
        for file_path, unified_symbols in path_to_unified_symbols.items():
            symbols_in_file = []
            for us in unified_symbols:
                symbol = LanguageServerSymbol(us)
                symbols_in_file.append(
                    symbol.to_dict(
                        depth=depth,
                        kind=True,
                        include_relative_path=False,
                        location=False,
                        child_inclusion_predicate=child_inclusion_predicate,
                    )
                )
            result[file_path] = symbols_in_file

        return result


class JetBrainsSymbol(Symbol):
    class SymbolDict(TypedDict):
        name_path: str
        relative_path: str
        type: str
        text_range: NotRequired[dict]
        body: NotRequired[str]
        children: NotRequired[list["JetBrainsSymbol.SymbolDict"]]

    def __init__(self, symbol_dict: SymbolDict, project: Project) -> None:
        """
        :param symbol_dict: dictionary as returned by the JetBrains plugin client.
        """
        self._project = project
        self._dict = symbol_dict
        self._cached_file_content: str | None = None
        self._cached_body_start_position: PositionInFile | None = None
        self._cached_body_end_position: PositionInFile | None = None

    def _tostring_includes(self) -> list[str]:
        return []

    def _tostring_additional_entries(self) -> dict[str, Any]:
        return dict(name_path=self.get_name_path(), relative_path=self.get_relative_path(), type=self._dict["type"])

    def get_name_path(self) -> str:
        return self._dict["name_path"]

    def get_relative_path(self) -> str:
        return self._dict["relative_path"]

    def get_file_content(self) -> str:
        if self._cached_file_content is None:
            path = os.path.join(self._project.project_root, self.get_relative_path())
            with open(path, encoding=self._project.project_config.encoding) as f:
                self._cached_file_content = f.read()
        return self._cached_file_content

    def is_position_in_file_available(self) -> bool:
        return "text_range" in self._dict

    def get_body_start_position(self) -> PositionInFile | None:
        if not self.is_position_in_file_available():
            return None
        if self._cached_body_start_position is None:
            pos = self._dict["text_range"]["start_pos"]
            line, col = pos["line"], pos["col"]
            self._cached_body_start_position = PositionInFile(line=line, col=col)
        return self._cached_body_start_position

    def get_body_end_position(self) -> PositionInFile | None:
        if not self.is_position_in_file_available():
            return None
        if self._cached_body_end_position is None:
            pos = self._dict["text_range"]["end_pos"]
            line, col = pos["line"], pos["col"]
            self._cached_body_end_position = PositionInFile(line=line, col=col)
        return self._cached_body_end_position

    def is_neighbouring_definition_separated_by_empty_line(self) -> bool:
        # NOTE: Symbol types cannot really be differentiated, because types are not handled in a language-agnostic way.
        return False
