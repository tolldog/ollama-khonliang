"""Tests for khonliang MCP server."""

import pytest

try:
    from mcp.server.fastmcp import FastMCP

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from khonliang.gateway.blackboard import Blackboard

pytestmark = pytest.mark.skipif(not HAS_MCP, reason="mcp package not installed")


@pytest.fixture
def blackboard():
    return Blackboard(default_ttl=60)


@pytest.fixture
def server_no_components():
    from khonliang.mcp.server import KhonliangMCPServer

    return KhonliangMCPServer()


@pytest.fixture
def server_with_blackboard(blackboard):
    from khonliang.mcp.server import KhonliangMCPServer

    return KhonliangMCPServer(blackboard=blackboard)


class TestKhonliangMCPServer:
    def test_instantiation_no_components(self, server_no_components):
        assert server_no_components.knowledge_store is None
        assert server_no_components.blackboard is None
        assert server_no_components.roles == {}

    def test_create_app_returns_fastmcp(self, server_no_components):
        app = server_no_components.create_app()
        assert isinstance(app, FastMCP)

    def test_create_app_with_blackboard(self, server_with_blackboard):
        app = server_with_blackboard.create_app()
        assert isinstance(app, FastMCP)

    def test_app_has_name(self, server_no_components):
        app = server_no_components.create_app()
        assert app.name == "khonliang"


class TestBlackboardTools:
    @pytest.mark.asyncio
    async def test_post_and_read(self, server_with_blackboard, blackboard):
        app = server_with_blackboard.create_app()
        tools = {t.name: t for t in await app.list_tools()}

        assert "blackboard_post" in tools
        assert "blackboard_read" in tools
        assert "blackboard_context" in tools

    def test_blackboard_direct_operations(self, blackboard):
        blackboard.post("agent1", "findings", "key1", "test content")
        entries = blackboard.read("findings")
        assert "key1" in entries
        assert entries["key1"] == "test content"

    def test_blackboard_context(self, blackboard):
        blackboard.post("agent1", "findings", "key1", "content1")
        blackboard.post("agent2", "findings", "key2", "content2")
        ctx = blackboard.build_context(sections=["findings"])
        assert "key1" in ctx
        assert "key2" in ctx
        assert "agent1" in ctx

    def test_blackboard_sections_resource(self, blackboard):
        blackboard.post("agent1", "sec1", "k1", "v1")
        blackboard.post("agent1", "sec2", "k2", "v2")
        assert "sec1" in blackboard.sections
        assert "sec2" in blackboard.sections


class TestToolRegistration:
    @pytest.mark.asyncio
    async def test_only_catalog_when_no_components(self, server_no_components):
        app = server_no_components.create_app()
        tools = await app.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "catalog"

    @pytest.mark.asyncio
    async def test_catalog_lists_registered_tools(self, server_with_blackboard):
        app = server_with_blackboard.create_app()
        result = await app.call_tool("catalog", {"detail": "brief"})
        text = result[0].text if hasattr(result[0], "text") else str(result[0])
        assert "blackboard_post" in text
        assert "blackboard_read" in text
        assert "GUIDES" in text

    @pytest.mark.asyncio
    async def test_catalog_compact_mode(self, server_with_blackboard):
        app = server_with_blackboard.create_app()
        result = await app.call_tool("catalog", {"detail": "compact"})
        text = result[0].text if hasattr(result[0], "text") else str(result[0])
        assert "tools=" in text
        assert "guides=" in text

    @pytest.mark.asyncio
    async def test_blackboard_tools_registered(self, server_with_blackboard):
        app = server_with_blackboard.create_app()
        tool_names = {t.name for t in await app.list_tools()}
        assert "blackboard_post" in tool_names
        assert "blackboard_read" in tool_names
        assert "blackboard_context" in tool_names

    @pytest.mark.asyncio
    async def test_knowledge_tools_not_registered_without_store(
        self, server_with_blackboard
    ):
        app = server_with_blackboard.create_app()
        tool_names = {t.name for t in await app.list_tools()}
        assert "knowledge_search" not in tool_names


class TestResourceRegistration:
    @pytest.mark.asyncio
    async def test_blackboard_resource_registered(self, server_with_blackboard):
        app = server_with_blackboard.create_app()
        resources = await app.list_resources()
        uris = {str(r.uri) for r in resources}
        assert "blackboard://sections" in uris

    @pytest.mark.asyncio
    async def test_no_resources_without_components(self, server_no_components):
        app = server_no_components.create_app()
        resources = await app.list_resources()
        assert len(resources) == 0
