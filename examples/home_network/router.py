"""
Home network router — routes queries to network-aware roles.
"""

from khonliang.roles.router import BaseRouter


class HomeNetworkRouter(BaseRouter):
    """
    Routes home network queries.

    Priority:
      1. Device monitoring: security-related, change detection
      2. Network info: general questions about devices/services
      3. Fallback: network_info (most questions are informational)
    """

    def __init__(self):
        super().__init__(fallback_role="network_info")

        # Security / monitoring queries
        self.register_keywords(
            [
                "new device",
                "unknown device",
                "suspicious",
                "changed",
                "missing",
                "intruder",
                "unauthorized",
                "security",
                "baseline",
                "monitor",
                "alert",
            ],
            "device_monitor",
        )

        # General network queries
        self.register_keywords(
            [
                "what's on",
                "devices",
                "services",
                "online",
                "offline",
                "printer",
                "homekit",
                "apple tv",
                "chromecast",
                "speaker",
                "camera",
                "thermostat",
                "network",
                "ip address",
                "mac address",
                "scan",
            ],
            "network_info",
        )
