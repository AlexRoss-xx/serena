"""Pascal language server integration (pasls) for Serena / SolidLSP."""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import threading
from typing import Any, Final, cast

from overrides import override

from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_config import LanguageServerConfig
from solidlsp.lsp_protocol_handler.lsp_types import InitializeParams
from solidlsp.lsp_protocol_handler.server import ProcessLaunchInfo
from solidlsp.settings import SolidLSPSettings

log = logging.getLogger(__name__)

_PASLS_ENV_VAR: Final[str] = "SERENA_PASLS_PATH"


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
        max_dirs = 1000  # Increased limit for large Delphi projects like Profile

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

        # Dynamically scan for source directories to set as search paths
        # This avoids the need for manual configuration files like .fp-params
        source_dirs = self._scan_for_source_dirs(repository_absolute_path)

        # Start with Delphi compatibility mode
        fpc_options = ["-Mdelphi"]

        # Add all discovered source directories as unit and include paths
        for src_dir in sorted(source_dirs):  # Sort for consistent ordering
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
        
        initialize_params: dict[str, Any] = {
            "locale": "en",
            "initializationOptions": {
                # Inject the discovered paths as Free Pascal Compiler options
                "fpcOptions": fpc_options,
                # Ensure the LS knows we want it to parse the program
                "program": "",
                # Enable SQLite symbol database for production performance
                "symbolDatabase": symbol_db_path,
                # Disable workspace symbol scanning during initialization to avoid timeout
                # on large projects. Database will build lazily on-demand.
                "workspaceSymbols": False,
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
