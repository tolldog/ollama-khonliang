"""
Network scanner — discovers devices and services on the local network.

Uses:
- avahi-browse (mDNS/Bonjour) for service discovery
- arp-scan or ip neigh for device enumeration

All commands are optional — missing tools are handled gracefully.
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class NetworkDevice:
    """A device discovered on the local network."""

    ip: str
    mac: str = ""
    hostname: str = ""
    vendor: str = ""
    services: List[str] = field(default_factory=list)
    source: str = ""  # "arp", "mdns", "ip_neigh"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ip": self.ip,
            "mac": self.mac,
            "hostname": self.hostname,
            "vendor": self.vendor,
            "services": self.services,
            "source": self.source,
        }


@dataclass
class MdnsService:
    """A service discovered via mDNS/Bonjour."""

    name: str
    service_type: str
    hostname: str
    ip: str = ""
    port: int = 0
    txt: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "service_type": self.service_type,
            "hostname": self.hostname,
            "ip": self.ip,
            "port": self.port,
            "txt": self.txt,
        }


class NetworkScanner:
    """
    Discovers devices and services on the local network.

    Example:
        scanner = NetworkScanner()
        devices = scanner.scan_devices()
        services = scanner.scan_services()
        summary = scanner.get_network_summary()
    """

    def scan_devices(self) -> List[NetworkDevice]:
        """Scan for devices using ip neigh (ARP table)."""
        devices = []
        try:
            result = subprocess.run(
                ["ip", "neigh", "show"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    ip = parts[0]
                    mac = parts[4] if parts[4] != "FAILED" else ""
                    state = parts[-1]
                    if state in ("REACHABLE", "STALE", "DELAY"):
                        devices.append(
                            NetworkDevice(
                                ip=ip, mac=mac, source="ip_neigh"
                            )
                        )
        except FileNotFoundError:
            logger.debug("ip command not found")
        except subprocess.TimeoutExpired:
            logger.warning("ip neigh timed out")
        except Exception as e:
            logger.warning(f"Device scan failed: {e}")
        return devices

    def scan_services(self, timeout: int = 5) -> List[MdnsService]:
        """Scan for mDNS/Bonjour services using avahi-browse."""
        services = []
        try:
            result = subprocess.run(
                [
                    "avahi-browse",
                    "-a",
                    "-t",
                    "-r",
                    "-p",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            for line in result.stdout.strip().split("\n"):
                if not line or line.startswith("+"):
                    continue
                if line.startswith("="):
                    # Resolved service line
                    parts = line.split(";")
                    if len(parts) >= 9:
                        services.append(
                            MdnsService(
                                name=parts[3],
                                service_type=parts[4],
                                hostname=parts[6],
                                ip=parts[7],
                                port=int(parts[8]) if parts[8].isdigit() else 0,
                            )
                        )
        except FileNotFoundError:
            logger.debug("avahi-browse not found — install avahi-utils")
        except subprocess.TimeoutExpired:
            logger.debug("avahi-browse timed out (normal)")
        except Exception as e:
            logger.warning(f"Service scan failed: {e}")
        return services

    def get_network_summary(self) -> str:
        """
        Get a human-readable summary of the local network.

        Suitable for injection into an LLM prompt as context.
        """
        devices = self.scan_devices()
        services = self.scan_services()

        parts = []

        if devices:
            parts.append(f"Devices on local network ({len(devices)}):")
            for d in devices:
                line = f"  - {d.ip}"
                if d.mac:
                    line += f" ({d.mac})"
                if d.hostname:
                    line += f" — {d.hostname}"
                parts.append(line)
        else:
            parts.append("No devices found on local network.")

        if services:
            parts.append(f"\nmDNS services ({len(services)}):")
            for s in services:
                line = f"  - {s.name} ({s.service_type})"
                if s.hostname:
                    line += f" on {s.hostname}"
                if s.port:
                    line += f":{s.port}"
                parts.append(line)
        else:
            parts.append("\nNo mDNS services discovered.")

        return "\n".join(parts)

    def to_json(self) -> str:
        """Return scan results as JSON."""
        return json.dumps(
            {
                "devices": [d.to_dict() for d in self.scan_devices()],
                "services": [s.to_dict() for s in self.scan_services()],
            },
            indent=2,
        )
