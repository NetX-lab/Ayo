from typing import Dict, List, Optional

from Ayo.opt_pass.base_pass import OPT_Pass


class PassManager:
    """Manager for handling optimization passes

    Features:
    - Pass registration and management
    - Pass selection based on node properties
    """

    def __init__(self):
        self.passes: Dict[str, OPT_Pass] = {}

    def register_pass(self, opt_pass: OPT_Pass) -> None:
        """Register an optimization pass

        Args:
            opt_pass: Optimization pass to register
        """
        self.passes[opt_pass.name] = opt_pass

    def get_passes(self) -> List[OPT_Pass]:
        """Get all registered passes

        Returns:
            List of registered optimization passes
        """
        return list(self.passes.values())

    def get_enabled_passes(self) -> List[OPT_Pass]:
        """Get all enabled passes

        Returns:
            List of enabled optimization passes
        """
        return [p for p in self.passes.values() if p.is_enabled()]

    def get_pass(self, name: str) -> Optional[OPT_Pass]:
        """Get a specific pass by name

        Args:
            name: Name of the pass to retrieve

        Returns:
            The requested pass or None if not found
        """
        return self.passes.get(name)

    def enable_pass(self, name: str) -> None:
        """Enable a specific pass

        Args:
            name: Name of the pass to enable
        """
        if name in self.passes:
            self.passes[name].enable()

    def disable_pass(self, name: str) -> None:
        """Disable a specific pass

        Args:
            name: Name of the pass to disable
        """
        if name in self.passes:
            self.passes[name].disable()
