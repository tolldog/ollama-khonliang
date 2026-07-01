"""Custom guides (added via add_guide) must categorize as `meta`.

The catalog lists a `meta` tool once under GUIDES and suppresses it from OTHER.
Before the fix _infer_category only knew the default guide names, so any custom
guide (research_guide, genealogy_guide, ...) double-listed in brief/full catalog.
"""

from khonliang.mcp.server import KhonliangMCPServer


def test_default_guide_is_meta():
    s = KhonliangMCPServer()
    assert s._infer_category("coding_guide") == "meta"


def test_custom_guide_is_meta_after_add_guide():
    s = KhonliangMCPServer()
    assert s._infer_category("genealogy_guide") == "other"  # not yet registered
    s.add_guide("genealogy_guide", "domain guide")
    assert s._infer_category("genealogy_guide") == "meta"   # the fix


def test_non_guide_tools_keep_their_category():
    s = KhonliangMCPServer()
    s.add_guide("genealogy_guide", "domain guide")
    assert s._infer_category("knowledge_search") == "knowledge"
    assert s._infer_category("triple_add") == "triples"
    assert s._infer_category("mystery_tool") == "other"
