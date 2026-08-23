from knowledge_engineering.relationship_discovery import RelationshipDiscoveryAgent, discover_relationships


def _canonical():
    return {
        "schema_version": "1.0",
        "artifacts": [
            {"id": "artifact:a", "name": "SOURCE_A"},
            {"id": "artifact:b", "name": "SOURCE_B"},
        ],
        "entities": [
            {"id": "entity:a", "name": "customer_id", "artifact_id": "artifact:a"},
            {"id": "entity:b", "name": "customer_id", "artifact_id": "artifact:b"},
            {"id": "entity:c", "name": "policy_id", "artifact_id": "artifact:b"},
        ],
        "relationships": [],
        "reference_matches": [],
    }


def test_reference_match_is_promoted_when_unambiguous_and_high_confidence():
    canonical = _canonical()
    canonical["reference_matches"] = [
        {
            "reference_entity_id": "entity:a",
            "candidate_entity_ids": ["entity:b"],
            "normalized_name": "customer_id",
            "confidence": 0.92,
        }
    ]

    result = discover_relationships(canonical)

    assert result["summary"]["cross_artifact_discovered"] == 1
    relationship = result["cross_artifact_relationships"][0]
    assert relationship["relationship"] == "REFERENCES"
    assert relationship["validation_status"] == "SUPPORTED"
    assert relationship["source_artifact_id"] == "artifact:a"
    assert relationship["target_artifact_id"] == "artifact:b"


def test_ambiguous_reference_remains_unverified():
    canonical = _canonical()
    canonical["entities"].append(
        {"id": "entity:d", "name": "customer_id", "artifact_id": "artifact:b"}
    )
    canonical["reference_matches"] = [
        {
            "reference_entity_id": "entity:a",
            "candidate_entity_ids": ["entity:b", "entity:d"],
            "normalized_name": "customer_id",
            "confidence": 0.95,
        }
    ]

    result = discover_relationships(canonical)

    assert result["summary"]["cross_artifact_discovered"] == 2
    assert all(
        item["validation_status"] == "UNVERIFIED"
        for item in result["cross_artifact_relationships"]
    )


def test_explicit_cross_artifact_relationship_is_supported_with_endpoint_evidence():
    canonical = _canonical()
    canonical["relationships"] = [
        {
            "id": "rel:1",
            "source_entity_id": "entity:a",
            "target_entity_id": "entity:b",
            "relationship_type": "MAPS_TO",
            "reconciliation_status": "DERIVED",
            "reconciled_confidence": 0.88,
        }
    ]

    result = discover_relationships(canonical)

    relationship = result["relationships"][0]
    assert relationship["relationship"] == "MAPS_TO"
    assert relationship["validation_status"] == "SUPPORTED"
    assert relationship["source_artifact_id"] == "artifact:a"
    assert relationship["target_artifact_id"] == "artifact:b"


def test_agent_exposes_discovery_metadata_and_safety_contract():
    result = RelationshipDiscoveryAgent().run(_canonical())

    assert result["agent"]["name"] == "relationship_discovery"
    assert result["agent"]["artifact_specific_hardcoding"] is False
    assert result["safety"]["source_modified"] is False
    assert result["safety"]["unsupported_inferences_are_unverified"] is True
