"""Pascal language server integration (pasls) for Serena / SolidLSP."""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Final, cast

from overrides import override

from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_config import LanguageServerConfig
from solidlsp.lsp_protocol_handler.lsp_types import InitializeParams
from solidlsp.lsp_protocol_handler.server import ProcessLaunchInfo
from solidlsp.settings import SolidLSPSettings

log = logging.getLogger(__name__)

_PASLS_ENV_VAR: Final[str] = "SERENA_PASLS_PATH"
_PASCAL_PROGRAM_ENV_VAR: Final[str] = "SERENA_PASCAL_PROGRAM"


class PascalLanguageServer(SolidLanguageServer):
    """
    Language-server wrapper for the Free Pascal / Object Pascal language server `pasls`.

    Expected server executable name:
      * `pasls.exe` on Windows
      * `pasls` on Unix-like systems

    How Serena finds the executable (in this order):

      1. Environment variable SERENA_PASLS_PATH pointing to the full path of pasls(.exe)
      2. `pasls` / `pasls.exe` discovered on the system PATH
      3. A local `pascal-language-server-genericptr` clone inside the Serena repo:
         - <serena-root>/pascal-language-server-genericptr/lib/*/pasls(.exe)

    If nothing is found, an informative RuntimeError is raised.

    Note: The enhanced fork uses InitWithEnvironmentVariables() to find FPC from PATH.
    Make sure Free Pascal Compiler is in your system PATH.
    """

    def __init__(
        self,
        config: LanguageServerConfig,
        repository_root_path: str,
        solidlsp_settings: SolidLSPSettings,
    ):
        pasls_path = self._locate_pasls_executable()

        # Support explicitly setting the path to FPC (compiler) via SERENA_FPC_PATH
        # This allows users to enforce a specific compiler version, which is critical for parsing
        env = os.environ.copy()
        fpc_path = os.environ.get("SERENA_FPC_PATH")
        if fpc_path:
            # Prepend to PATH so it takes precedence over system installed versions
            env["PATH"] = fpc_path + os.pathsep + env.get("PATH", "")
            log.info(f"Using explicitly configured FPC path: {fpc_path}")

        # CodeTools (used by pasls) relies on compiler/environment variables on many setups,
        # especially on Windows: PP, FPCDIR, LAZARUSDIR, FPCTARGET, FPCTARGETCPU.
        # Other clients (e.g. pasls-vscode / Emacs integrations) commonly set these.
        # If missing, CodeTools may fail to resolve includes (e.g. {$I UDefs.inc}) and
        # cross-file symbol operations become unreliable.
        if os.name == "nt":
            self._ensure_codetools_env_windows(env)

        super().__init__(
            config,
            repository_root_path,
            ProcessLaunchInfo(cmd=pasls_path, cwd=repository_root_path, env=env),
            "pascal",
            solidlsp_settings,
        )

        # Set generous timeout for Pascal on large projects (e.g., Profile with 1000+ files)
        # Large Delphi codebases with type libraries and complex generics can take time to parse
        # Initial workspace indexing for symbol database can take 10-15 minutes on very large projects
        self.set_request_timeout(1200.0)  # 20 minutes for initialization and individual LSP requests

        self.server_ready = threading.Event()
        self.request_id = 0
        if os.name == "nt":
            log.info("Pascal LS Windows URI workaround enabled (file:/// -> file://)")

    @staticmethod
    def _ensure_codetools_env_windows(env: dict[str, str]) -> None:
        """
        Best-effort setup for Lazarus/FreePascal environment variables required by CodeTools.

        Precedence:
        - If the user already defined PP/FPCDIR/LAZARUSDIR, keep them.
        - Otherwise, try to auto-detect Lazarus in common locations (C:\\lazarus).

        Users can also provide explicit overrides:
        - SERENA_PASCAL_PP, SERENA_PASCAL_FPCDIR, SERENA_PASCAL_LAZARUSDIR,
          SERENA_PASCAL_FPCTARGET, SERENA_PASCAL_FPCTARGETCPU
        """

        def _set_if_missing(key: str, value: str) -> None:
            if not env.get(key) and value:
                env[key] = value

        def _infer_target_from_pp(pp_path: str) -> tuple[str, str] | None:
            """
            Infer FPCTARGET/FPCTARGETCPU from compiler executable name when only one toolchain is installed.
            Common on Windows setups where only i386-win32 exists (ppc386.exe).
            """
            if not pp_path:
                return None
            name = Path(pp_path).name.lower()
            # FPC compiler names:
            # - ppc386.exe -> i386 / win32
            # - ppcx64.exe / ppcx86_64.exe -> x86_64 / win64
            if "ppc386" in name:
                return ("win32", "i386")
            if "ppcx64" in name or "ppcx86_64" in name or "ppcx86-64" in name:
                return ("win64", "x86_64")
            return None

        # Explicit overrides (project/user can set these)
        _set_if_missing("PP", env.get("SERENA_PASCAL_PP", ""))
        _set_if_missing("FPCDIR", env.get("SERENA_PASCAL_FPCDIR", ""))
        _set_if_missing("LAZARUSDIR", env.get("SERENA_PASCAL_LAZARUSDIR", ""))
        _set_if_missing("FPCTARGET", env.get("SERENA_PASCAL_FPCTARGET", ""))
        _set_if_missing("FPCTARGETCPU", env.get("SERENA_PASCAL_FPCTARGETCPU", ""))

        if env.get("PP") and env.get("FPCDIR") and env.get("LAZARUSDIR"):
            # If caller didn't specify target, infer from PP (most reliable on "only i386-win32 installed").
            if not env.get("FPCTARGET") or not env.get("FPCTARGETCPU"):
                inferred = _infer_target_from_pp(env.get("PP", ""))
                if inferred:
                    _set_if_missing("FPCTARGET", inferred[0])
                    _set_if_missing("FPCTARGETCPU", inferred[1])
            # Back-compat defaults
            _set_if_missing("FPCTARGET", "win32")
            _set_if_missing("FPCTARGETCPU", "i386")
            return

        lazarus_dir = Path(env.get("LAZARUSDIR", "")).expanduser()
        if not lazarus_dir.exists():
            # common default
            lazarus_dir = Path(r"C:\lazarus")

        if lazarus_dir.exists():
            _set_if_missing("LAZARUSDIR", str(lazarus_dir))
            fpc_root = lazarus_dir / "fpc"
            # pick latest version directory if present
            fpc_version_dir: Path | None = None
            if fpc_root.exists():
                versions = [p for p in fpc_root.iterdir() if p.is_dir()]
                # Prefer semver-ish name sorting
                versions.sort(key=lambda p: p.name)
                if versions:
                    fpc_version_dir = versions[-1]

            if fpc_version_dir:
                _set_if_missing("FPCDIR", str(fpc_version_dir))
                ppc = fpc_version_dir / "bin" / "i386-win32" / "ppc386.exe"
                if ppc.exists():
                    _set_if_missing("PP", str(ppc))
                    # Ensure compiler dir is on PATH as well (helps CodeToolsOptions.InitWithEnvironmentVariables)
                    env["PATH"] = str(ppc.parent) + os.pathsep + env.get("PATH", "")
                    # If target vars are missing, infer them from the compiler we just selected.
                    if not env.get("FPCTARGET") or not env.get("FPCTARGETCPU"):
                        inferred = _infer_target_from_pp(str(ppc))
                        if inferred:
                            _set_if_missing("FPCTARGET", inferred[0])
                            _set_if_missing("FPCTARGETCPU", inferred[1])
                    _set_if_missing("FPCTARGET", "win32")
                    _set_if_missing("FPCTARGETCPU", "i386")

    def _try_load_delphi_dproj_paths(self, repository_absolute_path: str) -> tuple[list[str], list[str]]:
        """
        Best-effort extraction of Delphi build configuration from a .dproj adjacent to the configured program.

        For Delphi codebases (like Profile), semantic resolution in CodeTools depends on matching the
        *real* UnitSearchPath/Defines from the Delphi project. We extract:
        - DCC_UnitSearchPath (as directories)
        - DCC_Define (as -d defines)
        """
        cfg_program = str(self._custom_settings.get("program", "")).strip()
        env_program = os.environ.get(_PASCAL_PROGRAM_ENV_VAR, "").strip()
        program_spec = env_program or cfg_program
        if not program_spec:
            return ([], [])

        program_path = program_spec
        if not os.path.isabs(program_path):
            program_path = os.path.join(repository_absolute_path, program_path)
        program_path = os.path.abspath(program_path)
        if not os.path.isfile(program_path):
            return ([], [])

        dproj_path = os.path.splitext(program_path)[0] + ".dproj"
        if not os.path.isfile(dproj_path):
            return ([], [])

        try:
            tree = ET.parse(dproj_path)
            root = tree.getroot()
        except Exception as e:
            log.warning("Failed parsing Delphi .dproj (%s): %s", dproj_path, e)
            return ([], [])

        ns = {"msb": "http://schemas.microsoft.com/developer/msbuild/2003"}

        # Collect all occurrences (different property groups may append with $(DCC_...))
        unit_search_values: list[str] = []
        define_values: list[str] = []
        for el in root.findall(".//msb:DCC_UnitSearchPath", ns):
            if el.text:
                unit_search_values.append(el.text)
        for el in root.findall(".//msb:DCC_Define", ns):
            if el.text:
                define_values.append(el.text)

        base_dir = os.path.dirname(dproj_path)

        def _norm_dir(p: str) -> str:
            p = p.strip()
            if not p or "$(" in p:
                return ""
            if not os.path.isabs(p):
                p = os.path.join(base_dir, p)
            p = os.path.abspath(p)
            return p if os.path.isdir(p) else ""

        dirs: list[str] = []
        seen_dirs: set[str] = set()
        for raw in unit_search_values:
            for part in raw.split(";"):
                d = _norm_dir(part)
                if d and d not in seen_dirs:
                    seen_dirs.add(d)
                    dirs.append(d)

        defines: list[str] = []
        seen_def: set[str] = set()
        for raw in define_values:
            for part in raw.split(";"):
                part = part.strip()
                if not part or "$(" in part:
                    continue
                if part not in seen_def:
                    seen_def.add(part)
                    defines.append(part)

        log.info("Loaded Delphi .dproj config: dirs=%d defines=%d (%s)", len(dirs), len(defines), dproj_path)
        return (dirs, defines)

    @staticmethod
    @override
    def _determine_log_level(line: str) -> int:
        """
        pasls sometimes logs full JSON-RPC payload fragments (including entire document text) to stderr.

        This is extremely verbose and can contain words like "error"/"exception" inside the document text,
        which would otherwise be misclassified as a real error by the default heuristic.
        """
        line_lower = (line or "").lower()

        # If pasls dumps JSON-RPC payloads to stderr (common in verbose/debug builds),
        # do not treat these as errors even if they contain words like "exception" as part of symbol names.
        # Example: {"jsonrpc":"2.0","id":...,"result":[... "TCRThreadExceptionEvent" ...]}
        if '"jsonrpc"' in line_lower and (line_lower.lstrip().startswith("{") or line_lower.lstrip().startswith("[")):
            return logging.DEBUG

        # Some pasls builds print JSON fragments across multiple stderr lines (e.g. just `"uri": "file://..."`).
        # These may include filenames containing "Exception" and would otherwise be misclassified as errors.
        if '"uri"' in line_lower and "file://" in line_lower:
            return logging.DEBUG

        # Some builds print file reload telemetry like:
        # "Reloaded <path> in 729ms"
        # Treat as INFO (not ERROR) even if the path contains "Exception".
        if line_lower.lstrip().startswith("reloaded ") and " in " in line_lower and "ms" in line_lower:
            return logging.INFO

        # If the stderr line appears to include a JSON payload fragment with the `"text"` field (didOpen/didChange),
        # treat it as DEBUG to avoid log spam and false-positive "ERROR" entries.
        if '"text"' in line_lower and ("unit " in line_lower or "\\nunit " in line_lower or "interface" in line_lower):
            return logging.DEBUG

        # Fall back to the default heuristic.
        if "error" in line_lower or "exception" in line_lower or (line or "").startswith("E["):
            return logging.ERROR
        return logging.INFO

    @override
    def is_ignored_dirname(self, dirname: str) -> bool:
        """Extend default ignored directories with Pascal-specific build folders.

        This method filters out:
        - Common Pascal build/backup directories
        - Known large external library directory names

        NOTE: We deliberately do NOT exclude generic names like 'tools', 'test', 'components'
        because these are often valid source directories in user Delphi projects.
        Only exclude directories that are clearly build artifacts or known external libraries.
        """
        dirname_lower = dirname.lower()

        # Standard Pascal build/backup directories - these NEVER contain source code
        pascal_build_dirs = {
            "lib",
            "backup",
            "bak",
            "tmp",
            "__history",
            "output",
            "bin",
            "obj",
            "__recovery",
            "dcu",
            "exe",
            "debug",
            "release",
        }

        # Known large external/third-party directory names that typically contain
        # full copies of external libraries that should be excluded
        external_library_dirs = {
            # Lazarus IDE / FPC complete source trees
            "lazarus-trunk",
            "lazarus",
            "fpc",
            "freepascal",
            # macOS app bundles (not source code)
            "lazarus.app",
            "startlazarus.app",
            # Common third-party lib root directories (when included as full copies)
            "synapse",
            "indy",
            "jedi",
            "jcl",
            "jvcl",
        }

        all_ignored = pascal_build_dirs | external_library_dirs

        return super().is_ignored_dirname(dirname) or dirname_lower in all_ignored

    @override
    def _path_to_uri(self, absolute_file_path: str) -> str:
        """
        Pasls (genericptr / CGE fork) on Windows expects file URIs in the form `file://D:/...`
        (two slashes) and may return `null` for requests when given the standard `file:///D:/...`.
        """
        uri = super()._path_to_uri(absolute_file_path)
        if os.name == "nt" and uri.startswith("file:///"):
            return uri.replace("file:///", "file://", 1)
        return uri

    def _scan_for_source_dirs(self, root_path: str) -> set[str]:
        """
        Recursively find all directories containing Pascal source files (.pas, .pp, .inc).

        Returns a set of absolute directory paths that should be added as -Fu and -Fi paths.
        Limits the number of directories to prevent overwhelming the compiler with too many paths.

        This method specifically handles Delphi-style project structures where:
        - Include files (.inc) may be in parent directories
        - Units (.pas) may reference units from sibling directory trees
        """
        valid_extensions = {".pas", ".pp", ".inc"}
        source_dirs: set[str] = set()
        # Increased limit for large Delphi projects like Profile, but allow overriding
        # since some pasls builds become unstable with huge option lists.
        try:
            max_dirs = int(os.environ.get("SERENA_PASCAL_MAX_SOURCE_DIRS", "1000"))
        except ValueError:
            max_dirs = 1000

        for root, dirs, files in os.walk(root_path):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if not self.is_ignored_dirname(d)]

            # Check if this directory contains any valid source files
            has_source = any(os.path.splitext(f)[1].lower() in valid_extensions for f in files)
            if has_source:
                source_dirs.add(root)

            # Stop if we've hit the limit
            if len(source_dirs) >= max_dirs:
                log.warning(
                    f"Pascal source directory scan hit limit of {max_dirs} directories. "
                    "Some directories may be excluded. Consider adding large external libraries "
                    "to ignored_paths in .serena/project.yml"
                )
                break

        if source_dirs:
            log.info(f"Found {len(source_dirs)} Pascal source directories in {os.path.basename(root_path)}")
            # Log first few directories for debugging (helpful for verifying paths are found)
            sample_dirs = sorted(source_dirs)[:5]
            for d in sample_dirs:
                log.debug(f"  Sample source dir: {d}")

        return source_dirs

    def _get_initialize_params(self, repository_absolute_path: str) -> InitializeParams:
        """Build LSP initialize parameters for pasls with dynamic include paths."""
        root_uri = pathlib.Path(repository_absolute_path).as_uri()
        # Hack for enhanced pasls on Windows: remove one slash if it starts with file:///
        if os.name == "nt" and root_uri.startswith("file:///"):
            root_uri = root_uri.replace("file:///", "file://")

        # Start with Delphi compatibility mode
        fpc_options = ["-Mdelphi"]

        # If a `castle-pasls.ini` exists at the repo root, prefer to keep initializationOptions.fpcOptions minimal.
        # The genericptr fork can read this ini automatically and inject a large set of -Fu/-Fi paths itself.
        # Passing thousands of options through LSP initialization can make some pasls builds unstable.
        castle_ini = os.path.join(repository_absolute_path, "castle-pasls.ini")
        force_scan_from_cfg = bool(self._custom_settings.get("force_scan_source_dirs", False))
        use_castle_ini_only = (
            os.path.isfile(castle_ini)
            and not force_scan_from_cfg
            and os.environ.get("SERENA_PASCAL_FORCE_SCAN_SOURCE_DIRS", "").strip() not in {
            "1",
            "true",
            "True",
            }
        )

        if use_castle_ini_only:
            # Minimal set: allow resolving units relative to repo root; let castle-pasls.ini inject the rest.
            fpc_options.append(f"-Fu{repository_absolute_path}")
            fpc_options.append(f"-Fi{repository_absolute_path}")
            # Semantic boost: import Delphi .dproj UnitSearchPath/Defines next to the configured program.
            dproj_dirs, dproj_defines = self._try_load_delphi_dproj_paths(repository_absolute_path)
            for d in dproj_dirs:
                fpc_options.append(f"-Fu{d}")
                # Delphi UnitSearchPath is commonly used for both units and includes
                fpc_options.append(f"-Fi{d}")
                fpc_options.append(f"-I{d}")
            for define in dproj_defines:
                fpc_options.append(f"-d{define}")
            log.info(
                "castle-pasls.ini detected; using minimal fpcOptions "
                "(set SERENA_PASCAL_FORCE_SCAN_SOURCE_DIRS=1 or ls_specific_settings.pascal.force_scan_source_dirs=true to override)."
            )
            source_dirs: set[str] = set()
        else:
            # Dynamically scan for source directories to set as search paths
            # This avoids the need for manual configuration files like .fp-params
            source_dirs = self._scan_for_source_dirs(repository_absolute_path)

            # Add all discovered source directories as unit and include paths
            for src_dir in sorted(source_dirs):  # Sort for consistent ordering
                # CodeTools parses compiler options from a command-line-like string.
                # Unquoted paths containing spaces can break parsing (and lead to missing include/unit paths),
                # which in turn causes spurious "include file not found" / 0-reference results.
                # We skip such paths to keep the configuration robust.
                if " " in src_dir:
                    log.debug("Skipping Pascal source dir with spaces (CodeTools parsing hazard): %s", src_dir)
                    continue
                # Add as unit path (-Fu) and include path (-Fi)
                # We use absolute paths to be safe
                fpc_options.append(f"-Fu{src_dir}")
                fpc_options.append(f"-Fi{src_dir}")

            log.info(f"Pascal LSP configured for {os.path.basename(repository_absolute_path)} with {len(source_dirs)} source directories")

            # Log the fpcOptions at debug level for troubleshooting
            if source_dirs:
                log.debug(f"fpcOptions count: {len(fpc_options)}")

        # Enable persistent symbol database for faster symbol queries on large projects
        symbol_db_path = os.path.join(repository_absolute_path, ".serena", "cache", "pascal", "symbols.db")
        os.makedirs(os.path.dirname(symbol_db_path), exist_ok=True)
        
        def _env_flag(name: str, default: bool) -> bool:
            val = os.environ.get(name)
            if val is None:
                return default
            val = val.strip().lower()
            if val in ("1", "true", "yes", "y", "on"):
                return True
            if val in ("0", "false", "no", "n", "off"):
                return False
            return default

        # Workspace symbol support is a major performance lever for global symbol search, BUT some pasls builds
        # crash during `initialize` when this is enabled (native "Access violation"/terminated stdout reader).
        # For robustness we keep it OFF by default; enable explicitly via env var if your pasls build is stable.
        workspace_symbols_enabled = _env_flag("SERENA_PASCAL_WORKSPACE_SYMBOLS", False)

        # Option A: Configure a single "program root" (.dpr/.lpr) for accurate cross-file references.
        # pasls uses this as the start unit for its uses graph in textDocument/references.
        program_path = ""
        # Prefer LS-specific settings (persistent config) over env var.
        # Example serena_config.yml:
        #   ls_specific_settings:
        #     pascal:
        #       program: "Profile/Client/ProfileXE104.dpr"
        cfg_program = str(self._custom_settings.get("program", "")).strip()
        # Allow env var to override project/global config for ad-hoc experiments and debugging.
        env_program = os.environ.get(_PASCAL_PROGRAM_ENV_VAR, "").strip()
        program_spec = env_program or cfg_program
        if program_spec:
            # Allow either absolute or repo-relative path
            candidate = program_spec
            if not os.path.isabs(candidate):
                candidate = os.path.join(repository_absolute_path, candidate)
            candidate = os.path.abspath(candidate)
            if os.path.isfile(candidate):
                program_path = candidate
                log.info(f"Using Pascal program root: {program_path}")
            else:
                log.warning(
                    f"Pascal program root is set but file not found: {candidate}. "
                    "pasls references may be incomplete; set it to a valid .dpr/.lpr."
                )

        initialize_params: dict[str, Any] = {
            "locale": "en",
            "initializationOptions": {
                # Inject the discovered paths as Free Pascal Compiler options
                "fpcOptions": fpc_options,
                # Ensure the LS knows we want it to parse the program
                "program": program_path,
                # Enable SQLite symbol database for production performance
                "symbolDatabase": symbol_db_path,
                # Workspace symbol support is a major performance lever for global symbol search.
                # If you hit LS initialization issues on huge workspaces, set SERENA_PASCAL_WORKSPACE_SYMBOLS=0.
                "workspaceSymbols": workspace_symbols_enabled,
            },
            "capabilities": {
                "textDocument": {
                    "synchronization": {"didSave": True, "dynamicRegistration": True},
                    "definition": {"dynamicRegistration": True},
                    "implementation": {"dynamicRegistration": True},
                    "references": {"dynamicRegistration": True},
                    "documentSymbol": {
                        "dynamicRegistration": True,
                        "hierarchicalDocumentSymbolSupport": True,
                        "symbolKind": {"valueSet": list(range(1, 27))},
                    },
                    "hover": {
                        "dynamicRegistration": True,
                        "contentFormat": ["markdown", "plaintext"],
                    },
                    "completion": {
                        "dynamicRegistration": True,
                        "completionItem": {"snippetSupport": True},
                    },
                    "signatureHelp": {"dynamicRegistration": True},
                    "documentHighlight": {"dynamicRegistration": True},
                    "codeAction": {"dynamicRegistration": True},
                },
                "workspace": {
                    "workspaceFolders": True,
                    "didChangeConfiguration": {"dynamicRegistration": True},
                    "symbol": {"dynamicRegistration": True},
                    "executeCommand": {"dynamicRegistration": True},
                },
            },
            "processId": os.getpid(),
            "rootUri": root_uri,
            "workspaceFolders": [
                {
                    "uri": root_uri,
                    "name": os.path.basename(repository_absolute_path),
                }
            ],
        }

        return cast(InitializeParams, initialize_params)

    @classmethod
    def _locate_pasls_executable(cls) -> str:
        """Find the pasls executable using env var, PATH, and common local build paths."""
        exe_name = "pasls.exe" if os.name == "nt" else "pasls"
        patched_exe_name = "pasls_patched.exe" if os.name == "nt" else "pasls_patched"

        # 1. Explicit path from environment variable
        env_path = os.environ.get(_PASLS_ENV_VAR)
        if env_path:
            expanded = os.path.expanduser(env_path)
            if os.path.isfile(expanded):
                # If a patched build exists alongside the configured pasls, prefer it.
                # This avoids locking issues when replacing the original binary and lets us
                # ship hotfixes without changing user environment variables.
                expanded_path = pathlib.Path(expanded)
                patched_candidates = [
                    expanded_path.with_name(patched_exe_name),
                    # Common layout: <repo>/lib/<arch>/pasls.exe, patched at <repo>/pasls_patched.exe
                    expanded_path.parents[2] / patched_exe_name if len(expanded_path.parents) >= 3 else None,
                ]
                for candidate in patched_candidates:
                    if candidate is not None and candidate.is_file():
                        log.info(f"Using patched pasls at {candidate} (preferred over {_PASLS_ENV_VAR}={expanded})")
                        return str(candidate)

                log.info(f"Using pasls from {_PASLS_ENV_VAR}={expanded}")
                return expanded
            log.warning(f"{_PASLS_ENV_VAR} is set but file not found: {expanded}")

        # 2. Search on PATH
        from_path = shutil.which(exe_name)
        if from_path:
            log.info(f"Found pasls executable on PATH at {from_path}")
            return from_path

        # 3. Look for a sibling or nested pascal-language-server-genericptr checkout
        try:
            repo_root = pathlib.Path(__file__).resolve().parents[4]
        except IndexError:
            repo_root = pathlib.Path(__file__).resolve().parent

        # Candidates for the base of the genericptr repo
        search_roots = [
            repo_root / "pascal-language-server-genericptr",
            repo_root / "Intrahealth" / "SERENA" / "pascal-language-server-genericptr",
        ]

        candidates: list[pathlib.Path] = []

        for pls_root in search_roots:
            lib_root = pls_root / "lib"
            if lib_root.is_dir():
                for arch_dir in lib_root.iterdir():
                    if arch_dir.is_dir():
                        candidates.append(arch_dir / exe_name)
            # also allow a patched executable at the repo root
            candidates.append(pls_root / patched_exe_name)

        for candidate in candidates:
            if candidate.is_file():
                log.info(f"Found pasls in local checkout at {candidate}")
                return str(candidate)

        # Nothing found
        message = (
            "Pascal language server executable 'pasls' was not found.\\n\\n"
            "Make sure you have built pasls and either:\\n"
            f"  * add it to your PATH, or\\n"
            f"  * set {_PASLS_ENV_VAR} to the full path of pasls.exe\\n"
        )
        log.error(message)
        raise RuntimeError(message)

    def _start_server(self) -> None:
        """Start pasls, send initialize, and mark the server as ready."""

        def register_capability_handler(params: Any) -> None:
            return None

        def window_log_message(msg: Any) -> None:
            log.info(f"LSP (pasls) window/logMessage: {msg}")

        def do_nothing(_params: Any) -> None:
            return None

        self.server.on_request("client/registerCapability", register_capability_handler)
        self.server.on_notification("window/logMessage", window_log_message)
        self.server.on_notification("$/progress", do_nothing)
        self.server.on_notification("textDocument/publishDiagnostics", do_nothing)

        log.info("Starting Pascal language server (pasls) process")
        self.server.start()

        initialize_params = self._get_initialize_params(self.repository_root_path)

        log.info("Sending initialize request to Pascal language server and waiting for response")
        init_response = self.server.send.initialize(initialize_params)

        capabilities = init_response.get("capabilities", {})
        if "textDocumentSync" not in capabilities:
            log.warning("pasls did not report textDocumentSync capability; continuing anyway")
        if "completionProvider" not in capabilities:
            log.warning("pasls did not report completionProvider capability; completions may be limited")

        self.server.notify.initialized({})

        self.completions_available.set()
        self.server_ready.set()
        self.server_ready.wait()
