"""
Language server-related tools
"""

import os
import shutil
import subprocess
from collections.abc import Sequence
from copy import copy
from typing import Any

from serena.tools import (
    SUCCESS_RESULT,
    Tool,
    ToolMarkerSymbolicEdit,
    ToolMarkerSymbolicRead,
)
from serena.tools.tools_base import ToolMarkerOptional
from solidlsp.ls_types import SymbolKind


def _sanitize_symbol_dict(symbol_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize a symbol dictionary inplace by removing unnecessary information.
    """
    # We replace the location entry, which repeats line information already included in body_location
    # and has unnecessary information on column, by just the relative path.
    symbol_dict = copy(symbol_dict)
    s_relative_path = symbol_dict.get("location", {}).get("relative_path")
    if s_relative_path is not None:
        symbol_dict["relative_path"] = s_relative_path
    symbol_dict.pop("location", None)
    # also remove name, name_path should be enough
    symbol_dict.pop("name")
    return symbol_dict


class RestartLanguageServerTool(Tool, ToolMarkerOptional):
    """Restarts the language server, may be necessary when edits not through Serena happen."""

    def apply(self) -> str:
        """Use this tool only on explicit user request or after confirmation.
        It may be necessary to restart the language server if it hangs.
        """
        self.agent.reset_language_server_manager()
        return SUCCESS_RESULT


class GetSymbolsOverviewTool(Tool, ToolMarkerSymbolicRead):
    """
    Gets an overview of the top-level symbols defined in a given file.
    """

    def apply(self, relative_path: str, depth: int = 0, max_answer_chars: int = -1) -> str:
        """
        Use this tool to get a high-level understanding of the code symbols in a file.
        This should be the first tool to call when you want to understand a new file, unless you already know
        what you are looking for.

        :param relative_path: the relative path to the file to get the overview of
        :param depth: depth up to which descendants of top-level symbols shall be retrieved
            (e.g. 1 retrieves immediate children). Default 0.
        :param max_answer_chars: if the overview is longer than this number of characters,
            no content will be returned. -1 means the default value from the config will be used.
            Don't adjust unless there is really no other way to get the content required for the task.
        :return: a JSON object containing info about top-level symbols in the file
        """
        symbol_retriever = self.create_language_server_symbol_retriever()
        file_path = os.path.join(self.project.project_root, relative_path)

        # The symbol overview is capable of working with both files and directories,
        # but we want to ensure that the user provides a file path.
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File or directory {relative_path} does not exist in the project.")
        if os.path.isdir(file_path):
            raise ValueError(f"Expected a file path, but got a directory path: {relative_path}. ")
        result = symbol_retriever.get_symbol_overview(relative_path, depth=depth)[relative_path]
        result_json_str = self._to_json(result)
        return self._limit_length(result_json_str, max_answer_chars)


class FindSymbolTool(Tool, ToolMarkerSymbolicRead):
    """
    Performs a global (or local) search using the language server backend.
    """

    # noinspection PyDefaultArgument
    def apply(
        self,
        name_path_pattern: str,
        depth: int = 0,
        relative_path: str = "",
        include_body: bool = False,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        substring_matching: bool = False,
        max_answer_chars: int = -1,
    ) -> str:
        """
        Retrieves information on all symbols/code entities (classes, methods, etc.) based on the given name path pattern.
        The returned symbol information can be used for edits or further queries.
        Specify `depth > 0` to also retrieve children/descendants (e.g., methods of a class).

        A name path is a path in the symbol tree *within a source file*.
        For example, the method `my_method` defined in class `MyClass` would have the name path `MyClass/my_method`.
        If a symbol is overloaded (e.g., in Java), a 0-based index is appended (e.g. "MyClass/my_method[0]") to
        uniquely identify it.

        To search for a symbol, you provide a name path pattern that is used to match against name paths.
        It can be
         * a simple name (e.g. "method"), which will match any symbol with that name
         * a relative path like "class/method", which will match any symbol with that name path suffix
         * an absolute name path "/class/method" (absolute name path), which requires an exact match of the full name path within the source file.
        Append an index `[i]` to match a specific overload only, e.g. "MyClass/my_method[1]".

        :param name_path_pattern: the name path matching pattern (see above)
        :param depth: depth up to which descendants shall be retrieved (e.g. use 1 to also retrieve immediate children;
            for the case where the symbol is a class, this will return its methods).
            Default 0.
        :param relative_path: Optional. Restrict search to this file or directory. If None, searches entire codebase.
            If a directory is passed, the search will be restricted to the files in that directory.
            If a file is passed, the search will be restricted to that file.
            If you have some knowledge about the codebase, you should use this parameter, as it will significantly
            speed up the search as well as reduce the number of results.
        :param include_body: If True, include the symbol's source code. Use judiciously.
        :param include_kinds: Optional. List of LSP symbol kind integers to include. (e.g., 5 for Class, 12 for Function).
            Valid kinds: 1=file, 2=module, 3=namespace, 4=package, 5=class, 6=method, 7=property, 8=field, 9=constructor, 10=enum,
            11=interface, 12=function, 13=variable, 14=constant, 15=string, 16=number, 17=boolean, 18=array, 19=object,
            20=key, 21=null, 22=enum member, 23=struct, 24=event, 25=operator, 26=type parameter.
            If not provided, all kinds are included.
        :param exclude_kinds: Optional. List of LSP symbol kind integers to exclude. Takes precedence over `include_kinds`.
            If not provided, no kinds are excluded.
        :param substring_matching: If True, use substring matching for the last element of the pattern, such that
            "Foo/get" would match "Foo/getValue" and "Foo/getData".
        :param max_answer_chars: Max characters for the JSON result. If exceeded, no content is returned.
            -1 means the default value from the config will be used.
        :return: a list of symbols (with locations) matching the name.
        """
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None
        symbol_retriever = self.create_language_server_symbol_retriever()
        symbols = symbol_retriever.find(
            name_path_pattern,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
            substring_matching=substring_matching,
            within_relative_path=relative_path,
        )
        # Lazily load bodies only when explicitly requested.
        # We avoid eager body extraction during indexing because it is expensive.
        if include_body:
            for s in symbols:
                if s.body is not None:
                    continue
                rel = s.relative_path
                if rel is None:
                    continue
                try:
                    ls = symbol_retriever.get_language_server(rel)
                    s.symbol_root["body"] = ls.retrieve_symbol_body(s.symbol_root)
                except Exception:
                    # Best-effort: if body retrieval fails (unsupported symbol shape, etc.), still return symbol metadata.
                    pass
        symbol_dicts = [_sanitize_symbol_dict(s.to_dict(kind=True, location=True, depth=depth, include_body=include_body)) for s in symbols]
        result = self._to_json(symbol_dicts)
        return self._limit_length(result, max_answer_chars)


class FindIdentifierFastTool(Tool, ToolMarkerSymbolicRead):
    """
    Fast identifier search using ripgrep (rg) with a bounded candidate set.

    This is intended for cases where LSP-backed `find_symbol` is slow or unreliable
    (common for Pascal `const` identifiers like CID_*), and you just need a fast,
    correct "where does this identifier appear" answer.
    """

    def apply(
        self,
        identifier: str,
        relative_path: str = "",
        case_insensitive: bool = True,
        max_files: int = 50,
        max_matches_per_file: int = 20,
        include_globs: str = "*.{pas,pp,inc,dpr,lpr,dpk}",
        max_answer_chars: int = -1,
    ) -> str:
        """
        :param identifier: identifier string to search (literal match; no regex)
        :param relative_path: optional file/dir scope relative to project root
        :param case_insensitive: whether to match case-insensitively (recommended for Pascal)
        :param max_files: maximum number of files to scan/report (bounded for speed)
        :param max_matches_per_file: maximum matches per file (bounded for speed)
        :param include_globs: comma/brace-glob string passed to rg as --glob; default targets Pascal/Delphi sources
        :param max_answer_chars: max characters for JSON result (Serena output limiter)
        :return: JSON object containing candidate files and matches (file, line, column, text)
        """
        identifier = (identifier or "").strip()
        if not identifier:
            raise ValueError("identifier must be non-empty")

        root = self.get_project_root()
        scope_rel = relative_path.strip() if relative_path else "."
        scope_abs = os.path.join(root, scope_rel) if scope_rel else root
        if not os.path.exists(scope_abs):
            raise FileNotFoundError(f"Relative path {relative_path} does not exist.")

        max_files = max(1, int(max_files))
        max_matches_per_file = max(1, int(max_matches_per_file))

        # Prefer ripgrep.
        rg = shutil.which("rg")
        matches: list[dict[str, Any]] = []
        candidate_files: list[str] = []

        def _parse_rg_lines(text: str) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for line in (text or "").splitlines():
                # format: path:line:col:text
                parts = line.split(":", 3)
                if len(parts) != 4:
                    continue
                rel, ln, col, content = parts
                try:
                    out.append(
                        {
                            "relative_path": rel,
                            "line": int(ln),  # 1-based (rg)
                            "column": int(col),  # 1-based (rg)
                            "text": content,
                        }
                    )
                except Exception:
                    continue
            return out

        def _normalize_paths(lines: list[str]) -> list[str]:
            out: list[str] = []
            for ln in lines:
                p = (ln or "").strip()
                if not p:
                    continue
                # rg returns paths relative to cwd if invoked with cwd=root
                out.append(p)
                if len(out) >= max_files:
                    break
            return out

        # Build globs: allow "a,b,c" and "{a,b,c}" style.
        globs: list[str] = []
        for g in (include_globs or "").split(","):
            g = g.strip()
            if g:
                globs.append(g)

        if rg:
            # 1) Get candidate files quickly.
            cmd_files = [
                rg,
                "--files-with-matches",
                "--fixed-strings",
                "--no-messages",
                "--hidden",
                "--follow",
            ]
            if case_insensitive:
                cmd_files.append("--ignore-case")
            for g in globs:
                cmd_files.extend(["--glob", g])
            cmd_files.extend([identifier, scope_rel])
            proc_files = subprocess.run(
                cmd_files, cwd=root, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if proc_files.returncode in (0, 1):
                candidate_files = _normalize_paths(proc_files.stdout.splitlines())

            # 2) Collect line matches from those candidates (bounded).
            if candidate_files:
                cmd_hits = [
                    rg,
                    "--no-heading",
                    "--line-number",
                    "--column",
                    "--fixed-strings",
                    "--no-messages",
                    "--hidden",
                    "--follow",
                    f"--max-count={max_matches_per_file}",
                ]
                if case_insensitive:
                    cmd_hits.append("--ignore-case")
                cmd_hits.append(identifier)
                cmd_hits.extend(candidate_files)
                proc_hits = subprocess.run(
                    cmd_hits, cwd=root, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace"
                )
                if proc_hits.returncode in (0, 1):
                    matches = _parse_rg_lines(proc_hits.stdout)

        # 2) Fallback: git grep (tracked files only).
        if not candidate_files:
            git = shutil.which("git")
            if git:
                cmd = [
                    git,
                    "-C",
                    root,
                    "grep",
                    "-n",
                    "-I",
                    "--fixed-strings",
                    "--no-color",
                    "--",
                    identifier,
                    scope_rel,
                ]
                proc = subprocess.run(cmd, cwd=root, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if proc.returncode in (0, 1):
                    # format: path:line:text
                    for ln in (proc.stdout or "").splitlines():
                        parts = ln.split(":", 2)
                        if len(parts) != 3:
                            continue
                        rel, line_no, content = parts
                        # quick glob filter (extensions) since git grep doesn't filter by glob portably
                        if globs:
                            # best-effort: only apply extension filter for the default Pascal globs
                            _, ext = os.path.splitext(rel.lower())
                            if ext not in {".pas", ".pp", ".inc", ".dpr", ".lpr", ".dpk"}:
                                continue
                        try:
                            matches.append({"relative_path": rel, "line": int(line_no), "column": None, "text": content})
                        except Exception:
                            continue
                        if len(matches) >= (max_files * max_matches_per_file):
                            break
                    # derive candidates from matches
                    seen: set[str] = set()
                    for m in matches:
                        rp = m["relative_path"]
                        if rp not in seen:
                            seen.add(rp)
                            candidate_files.append(rp)
                            if len(candidate_files) >= max_files:
                                break

        result_obj = {
            "identifier": identifier,
            "scope": relative_path or "",
            "case_insensitive": case_insensitive,
            "candidate_files": candidate_files,
            "matches": matches,
        }
        return self._limit_length(self._to_json(result_obj), max_answer_chars)


class FindReferencingSymbolsTool(Tool, ToolMarkerSymbolicRead):
    """
    Finds symbols that reference the given symbol using the language server backend
    """

    # noinspection PyDefaultArgument
    def apply(
        self,
        name_path: str,
        relative_path: str,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        max_answer_chars: int = -1,
    ) -> str:
        """
        Finds references to the symbol at the given `name_path`. The result will contain metadata about the referencing symbols
        as well as a short code snippet around the reference.

        :param name_path: for finding the symbol to find references for, same logic as in the `find_symbol` tool.
        :param relative_path: the relative path to the file containing the symbol for which to find references.
            Note that here you can't pass a directory but must pass a file.
        :param include_kinds: same as in the `find_symbol` tool.
        :param exclude_kinds: same as in the `find_symbol` tool.
        :param max_answer_chars: same as in the `find_symbol` tool.
        :return: a list of JSON objects with the symbols referencing the requested symbol
        """
        include_body = False  # It is probably never a good idea to include the body of the referencing symbols
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None
        symbol_retriever = self.create_language_server_symbol_retriever()
        references_in_symbols = symbol_retriever.find_referencing_symbols(
            name_path,
            relative_file_path=relative_path,
            include_body=include_body,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
        )
        reference_dicts = []
        for ref in references_in_symbols:
            ref_dict = ref.symbol.to_dict(kind=True, location=True, depth=0, include_body=include_body)
            ref_dict = _sanitize_symbol_dict(ref_dict)
            if not include_body:
                ref_relative_path = ref.symbol.location.relative_path
                assert ref_relative_path is not None, f"Referencing symbol {ref.symbol.name} has no relative path, this is likely a bug."
                content_around_ref = self.project.retrieve_content_around_line(
                    relative_file_path=ref_relative_path, line=ref.line, context_lines_before=1, context_lines_after=1
                )
                ref_dict["content_around_reference"] = content_around_ref.to_display_string()
            reference_dicts.append(ref_dict)
        result = self._to_json(reference_dicts)
        return self._limit_length(result, max_answer_chars)


class ReplaceSymbolBodyTool(Tool, ToolMarkerSymbolicEdit):
    """
    Replaces the full definition of a symbol using the language server backend.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        r"""
        Replaces the body of the symbol with the given `name_path`.

        The tool shall be used to replace symbol bodies that have been previously retrieved
        (e.g. via `find_symbol`).
        IMPORTANT: Do not use this tool if you do not know what exactly constitutes the body of the symbol.

        :param name_path: for finding the symbol to replace, same logic as in the `find_symbol` tool.
        :param relative_path: the relative path to the file containing the symbol
        :param body: the new symbol body. The symbol body is the definition of a symbol
            in the programming language, including e.g. the signature line for functions.
            IMPORTANT: The body does NOT include any preceding docstrings/comments or imports, in particular.
        """
        code_editor = self.create_code_editor()
        code_editor.replace_body(
            name_path,
            relative_file_path=relative_path,
            body=body,
        )
        return SUCCESS_RESULT


class InsertAfterSymbolTool(Tool, ToolMarkerSymbolicEdit):
    """
    Inserts content after the end of the definition of a given symbol.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        """
        Inserts the given body/content after the end of the definition of the given symbol (via the symbol's location).
        A typical use case is to insert a new class, function, method, field or variable assignment.

        :param name_path: name path of the symbol after which to insert content (definitions in the `find_symbol` tool apply)
        :param relative_path: the relative path to the file containing the symbol
        :param body: the body/content to be inserted. The inserted code shall begin with the next line after
            the symbol.
        """
        code_editor = self.create_code_editor()
        code_editor.insert_after_symbol(name_path, relative_file_path=relative_path, body=body)
        return SUCCESS_RESULT


class InsertBeforeSymbolTool(Tool, ToolMarkerSymbolicEdit):
    """
    Inserts content before the beginning of the definition of a given symbol.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        """
        Inserts the given content before the beginning of the definition of the given symbol (via the symbol's location).
        A typical use case is to insert a new class, function, method, field or variable assignment; or
        a new import statement before the first symbol in the file.

        :param name_path: name path of the symbol before which to insert content (definitions in the `find_symbol` tool apply)
        :param relative_path: the relative path to the file containing the symbol
        :param body: the body/content to be inserted before the line in which the referenced symbol is defined
        """
        code_editor = self.create_code_editor()
        code_editor.insert_before_symbol(name_path, relative_file_path=relative_path, body=body)
        return SUCCESS_RESULT


class RenameSymbolTool(Tool, ToolMarkerSymbolicEdit):
    """
    Renames a symbol throughout the codebase using language server refactoring capabilities.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        new_name: str,
    ) -> str:
        """
        Renames the symbol with the given `name_path` to `new_name` throughout the entire codebase.
        Note: for languages with method overloading, like Java, name_path may have to include a method's
        signature to uniquely identify a method.

        :param name_path: name path of the symbol to rename (definitions in the `find_symbol` tool apply)
        :param relative_path: the relative path to the file containing the symbol to rename
        :param new_name: the new name for the symbol
        :return: result summary indicating success or failure
        """
        code_editor = self.create_code_editor()
        status_message = code_editor.rename_symbol(name_path, relative_file_path=relative_path, new_name=new_name)
        return status_message
