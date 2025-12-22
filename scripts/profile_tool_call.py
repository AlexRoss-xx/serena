import cProfile
import os
from pathlib import Path
from typing import Literal

from sensai.util import logging
from sensai.util.logging import LogTime
from sensai.util.profiling import profiled

from serena.agent import SerenaAgent
from serena.config.serena_config import SerenaConfig
from serena.tools import ActivateProjectTool
from serena.tools import FindSymbolTool

log = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.configure()

    # The profiler to use:
    # Use pyinstrument for hierarchical profiling output
    # Use cProfile to determine which functions take the most time overall (and use snakeviz to visualize)
    profiler: Literal["pyinstrument", "cprofile"] = "cprofile"

    project_path = Path(__file__).parent.parent  # Serena root

    serena_config = SerenaConfig.from_config_file()
    serena_config.log_level = logging.INFO
    serena_config.gui_log_window_enabled = False
    serena_config.web_dashboard = False

    # This script is timing-focused; suppress extremely verbose LSP stderr debug dumps.
    logging.getLogger("solidlsp.ls_handler").setLevel(logging.WARNING)

    agent = SerenaAgent(str(project_path), serena_config=serena_config)

    # wait for language server to be ready
    agent.execute_task(lambda: log.info("Language server is ready."))

    # Optional: activate an explicit target project (e.g. profile) via env var.
    target_project = os.environ.get("SERENA_PROFILE_PROJECT")
    if target_project:
        with LogTime(f"activate_project({target_project})"):
            agent.get_tool(ActivateProjectTool).apply(project=target_project)
        # wait for language server manager to finish initializing for the activated project
        agent.execute_task(lambda: log.info("Language server is ready (post-activation)."))

    def tool_call():
        """This is the function we want to profile."""
        # NOTE: We use apply (not apply_ex) to run the tool call directly on the main thread
        with LogTime("Tool call"):
            pattern = os.environ.get("SERENA_PROFILE_SYMBOL", "DQN")
            result = agent.get_tool(FindSymbolTool).apply(name_path_pattern=pattern, depth=0, relative_path="")
        log.info("Tool result:\n%s", result)

    if profiler == "pyinstrument":

        @profiled(log_to_file=True)
        def profiled_tool_call():
            tool_call()

        profiled_tool_call()

    elif profiler == "cprofile":
        cProfile.run("tool_call()", "tool_call.pstat")
