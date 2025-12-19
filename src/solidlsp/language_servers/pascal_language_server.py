"""Pascal language server integration (pasls) for Serena / SolidLSP."""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import threading
from typing import Final

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

        super().__init__(
            config,
            repository_root_path,
            ProcessLaunchInfo(cmd=pasls_path, cwd=repository_root_path),
            "pascal",
            solidlsp_settings,
        )

        self.server_ready = threading.Event()
        self.request_id = 0

    @override
    def is_ignored_dirname(self, dirname: str) -> bool:
        """Extend default ignored directories with Pascal-specific build folders."""
        return super().is_ignored_dirname(dirname) or dirname.lower() in {
            "lib", "backup", "bak", "tmp",
        }

    @classmethod
    def _locate_pasls_executable(cls) -> str:
        """Find the pasls executable using env var, PATH, and common local build paths."""
        exe_name = "pasls.exe" if os.name == "nt" else "pasls"

        # 1. Explicit path from environment variable
        env_path = os.environ.get(_PASLS_ENV_VAR)
        if env_path:
            expanded = os.path.expanduser(env_path)
            if os.path.isfile(expanded):
                log.info(f"Using pasls from {_PASLS_ENV_VAR}={expanded}")
                return expanded
            log.warning(f"{_PASLS_ENV_VAR} is set but file not found: {expanded}")

        # 2. Search on PATH
        from_path = shutil.which(exe_name)
        if from_path:
            log.info(f"Found pasls executable on PATH at {from_path}")
            return from_path

        # 3. Look for a sibling pascal-language-server-genericptr checkout
        try:
            repo_root = pathlib.Path(__file__).resolve().parents[4]
        except IndexError:
            repo_root = pathlib.Path(__file__).resolve().parent

        pls_root = repo_root / "pascal-language-server-genericptr"
        lib_root = pls_root / "lib"
        
        candidates: list[pathlib.Path] = []
        if lib_root.is_dir():
            for arch_dir in lib_root.iterdir():
                if arch_dir.is_dir():
                    candidates.append(arch_dir / exe_name)

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

    @staticmethod
    def _get_initialize_params(repository_absolute_path: str) -> InitializeParams:
        """Build LSP initialize parameters for pasls."""
        root_uri = pathlib.Path(repository_absolute_path).as_uri()
        # Hack for enhanced pasls on Windows: remove one slash if it starts with file:///
        # This is because the Pascal URIParser implementation on Windows seems to misinterpret
        # file:///D:/... as /D:/... (treating the first slash as root).
        # Sending file://D:/... seems to help it parse correctly as D:/...
        if os.name == 'nt' and root_uri.startswith('file:///'):
            root_uri = root_uri.replace('file:///', 'file://')

        initialize_params: InitializeParams = {
            "locale": "en",
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

        return initialize_params

    def _start_server(self) -> None:
        """Start pasls, send initialize, and mark the server as ready."""

        def register_capability_handler(params):
            return

        def window_log_message(msg):
            log.info(f"LSP (pasls) window/logMessage: {msg}")

        def do_nothing(_params):
            return

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
