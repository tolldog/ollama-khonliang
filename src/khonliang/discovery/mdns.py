"""
mDNS service advertiser.

Uses zeroconf to advertise services via mDNS/DNS-SD,
enabling discovery via hostname.local on the local network.

Requires: pip install ollama-khonliang[discovery]
"""

import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from zeroconf import ServiceInfo, Zeroconf
except ImportError:
    Zeroconf = None  # type: ignore[assignment,misc]
    ServiceInfo = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


@dataclass
class ServiceDefinition:
    """Definition of a service to advertise."""

    name: str
    port: int
    service_type: str = "_http._tcp.local."
    path: str = "/"
    version: str = "1.0"
    properties: Dict[str, str] = field(default_factory=dict)


class ServiceAdvertiser:
    """
    Advertises services via mDNS/DNS-SD.

    Allows services to be discovered on the local network using
    the configured hostname (default: khonliang.local).

    Example:
        >>> advertiser = ServiceAdvertiser(hostname="myapp")
        >>> advertiser.register_service(ServiceDefinition(
        ...     name="api",
        ...     port=8000,
        ...     path="/api/v1",
        ... ))
        >>> advertiser.start()
        >>> # Services now discoverable via myapp.local
        >>> advertiser.stop()
    """

    def __init__(
        self,
        hostname: str = "khonliang",
        enabled: bool = True,
        services: Optional[List[ServiceDefinition]] = None,
    ):
        self.hostname = hostname
        self.enabled = enabled
        self.services = services or []

        self._zeroconf: Optional[Any] = None
        self._registered_services: Dict[str, Any] = {}
        self._started = False

        if Zeroconf is None:
            logger.warning(
                "zeroconf library not installed. "
                "Install with: pip install ollama-khonliang[discovery]"
            )
            self.enabled = False

    def register_service(self, service: ServiceDefinition) -> None:
        self.services.append(service)
        logger.debug(f"Registered service: {service.name} on port {service.port}")

    def update_service_port(self, name: str, port: int) -> bool:
        for service in self.services:
            if service.name == name:
                service.port = port
                logger.debug(f"Updated {name} port to {port}")
                return True
        return False

    def _get_local_ip(self) -> str:
        for attempt in range(3):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.settimeout(2)
                    s.connect(("8.8.8.8", 80))
                    ip = s.getsockname()[0]
                if ip and ip != "127.0.0.1":
                    return ip
            except Exception as e:
                logger.debug(f"IP detection attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(1)

        logger.warning("Could not detect local IP, using 127.0.0.1")
        return "127.0.0.1"

    def _create_service_info(self, service: ServiceDefinition) -> Any:
        local_ip = self._get_local_ip()
        properties = {
            "path": service.path,
            "version": service.version,
            **service.properties,
        }
        service_name = f"{self.hostname}-{service.name}.{service.service_type}"

        return ServiceInfo(
            service.service_type,
            service_name,
            addresses=[socket.inet_aton(local_ip)],
            port=service.port,
            properties=properties,
            server=f"{self.hostname}.local.",
        )

    def start(self) -> bool:
        if not self.enabled:
            logger.info("mDNS advertisement disabled")
            return False

        if self._started:
            logger.warning("mDNS advertiser already started")
            return True

        try:
            self._zeroconf = Zeroconf()

            for service in self.services:
                registered = False
                for attempt in range(3):
                    try:
                        service_info = self._create_service_info(service)
                        self._zeroconf.register_service(service_info)
                        self._registered_services[service.name] = service_info
                        logger.info(
                            f"Registered mDNS service: {service.name} "
                            f"({self.hostname}.local:{service.port})"
                        )
                        registered = True
                        break
                    except Exception as e:
                        err_msg = str(e) if str(e) else type(e).__name__
                        if attempt < 2:
                            logger.debug(
                                f"mDNS registration attempt {attempt + 1} "
                                f"for {service.name} failed: {err_msg}"
                            )
                            time.sleep(0.5)
                        else:
                            logger.error(
                                f"Failed to register service {service.name}"
                                f" after 3 attempts: {err_msg}"
                            )

                if not registered:
                    logger.warning(
                        f"Service {service.name} not registered via mDNS"
                    )

            self._started = True
            logger.info(
                f"mDNS advertiser started - "
                f"{len(self._registered_services)} services "
                f"on {self.hostname}.local"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start mDNS advertiser: {e}")
            return False

    def stop(self) -> None:
        if not self._started or not self._zeroconf:
            return

        for name, service_info in self._registered_services.items():
            try:
                self._zeroconf.unregister_service(service_info)
                logger.debug(f"Unregistered mDNS service: {name}")
            except Exception as e:
                logger.warning(f"Error unregistering service {name}: {e}")

        try:
            self._zeroconf.close()
        except Exception as e:
            logger.warning(f"Error closing zeroconf: {e}")

        self._registered_services.clear()
        self._zeroconf = None
        self._started = False
        logger.info("mDNS advertiser stopped")

    def get_status(self) -> Dict:
        return {
            "enabled": self.enabled,
            "started": self._started,
            "hostname": f"{self.hostname}.local",
            "services": [
                {
                    "name": s.name,
                    "port": s.port,
                    "url": f"http://{self.hostname}.local:{s.port}{s.path}",
                }
                for s in self.services
            ],
            "registered_count": len(self._registered_services),
        }

    def __enter__(self) -> "ServiceAdvertiser":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
