"""
Home network roles — agents for local network awareness.

- NetworkInfoRole: answers questions about devices and services on the LAN
- DeviceMonitorRole: tracks device presence and reports changes
"""

from typing import Any, Dict, Optional

from examples.home_network.scanner import NetworkScanner
from khonliang.roles.base import BaseRole


class NetworkInfoRole(BaseRole):
    """
    Answers questions about the local network using live scan data.

    Injects a real-time network summary into the prompt context so the
    LLM can answer questions like "what's on my network?" or "is the
    printer online?"
    """

    def __init__(self, model_pool, scanner: Optional[NetworkScanner] = None, **kwargs):
        super().__init__(role="network_info", model_pool=model_pool, **kwargs)
        self.scanner = scanner or NetworkScanner()
        self._system_prompt = (
            "You are a home network assistant. You have access to a live scan "
            "of the local network showing connected devices and mDNS services. "
            "Answer questions about what's on the network, device status, and "
            "services. Be concise and helpful. If a device isn't in the scan "
            "data, say it appears to be offline or not discoverable."
        )

    def build_context(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        return self.scanner.get_network_summary()

    async def handle(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = self.build_context(message)
        prompt = (
            f"Current network state:\n{ctx}\n\n"
            f"User question: {message}\n\nAnswer:"
        )

        response, elapsed_ms = await self._timed_generate(
            prompt=prompt, system=self.system_prompt
        )
        return {
            "response": response.strip(),
            "metadata": {
                "role": self.role,
                "generation_time_ms": elapsed_ms,
                "devices_found": len(self.scanner.scan_devices()),
                "services_found": len(self.scanner.scan_services()),
            },
        }


class DeviceMonitorRole(BaseRole):
    """
    Monitors device presence and reports changes.

    Compares the current network scan against a known baseline to detect
    new, missing, or changed devices.
    """

    def __init__(self, model_pool, scanner: Optional[NetworkScanner] = None, **kwargs):
        super().__init__(role="device_monitor", model_pool=model_pool, **kwargs)
        self.scanner = scanner or NetworkScanner()
        self._known_devices: Dict[str, str] = {}  # ip -> mac
        self._system_prompt = (
            "You are a network security monitor. You've been given information "
            "about new, missing, and known devices on the local network. "
            "Summarize any changes and flag anything unusual (new unknown "
            "devices, devices that went offline). Be concise."
        )

    def snapshot_baseline(self) -> int:
        """Take a snapshot of current devices as the baseline."""
        devices = self.scanner.scan_devices()
        self._known_devices = {d.ip: d.mac for d in devices if d.mac}
        return len(self._known_devices)

    def detect_changes(self) -> Dict[str, Any]:
        """Compare current scan against baseline."""
        current = self.scanner.scan_devices()
        current_map = {d.ip: d.mac for d in current if d.mac}

        new_devices = [
            ip for ip in current_map if ip not in self._known_devices
        ]
        missing_devices = [
            ip for ip in self._known_devices if ip not in current_map
        ]
        changed_mac = [
            ip
            for ip in current_map
            if ip in self._known_devices
            and current_map[ip] != self._known_devices[ip]
        ]

        return {
            "new": new_devices,
            "missing": missing_devices,
            "changed_mac": changed_mac,
            "total_current": len(current_map),
            "total_baseline": len(self._known_devices),
        }

    async def handle(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        changes = self.detect_changes()

        change_summary = []
        if changes["new"]:
            change_summary.append(f"New devices: {', '.join(changes['new'])}")
        if changes["missing"]:
            change_summary.append(
                f"Missing devices: {', '.join(changes['missing'])}"
            )
        if changes["changed_mac"]:
            change_summary.append(
                f"MAC changed: {', '.join(changes['changed_mac'])}"
            )
        if not change_summary:
            change_summary.append("No changes detected since last baseline.")

        change_text = "\n".join(change_summary)
        prompt = (
            f"Network changes:\n{change_text}\n\n"
            f"Baseline: {changes['total_baseline']} devices, "
            f"Current: {changes['total_current']} devices.\n\n"
            f"User message: {message}\n\nAnalysis:"
        )

        response, elapsed_ms = await self._timed_generate(
            prompt=prompt, system=self.system_prompt
        )
        return {
            "response": response.strip(),
            "metadata": {
                "role": self.role,
                "generation_time_ms": elapsed_ms,
                "changes": changes,
            },
        }
