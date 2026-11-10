from src.utils.core import digest, path, read_json, stamp, write_json


def phase_report(cfg):
    def optional(filename):
        return read_json(filename) if filename.is_file() else None
    split = optional(path(cfg, "splits") / "split_manifest.json")
    audit = optional(path(cfg, "reports") / "audit.json")
    eda = optional(path(cfg, "reports") / "eda.json")
    embeddings = optional(path(cfg, "embeddings") / "manifest.json")
    embedding_progress = optional(path(cfg, "embeddings") / "progress.json")
    indexes = optional(path(cfg, "indexes") / "manifest.json")
    qc = optional(path(cfg, "reports") / "quality_control.json")
    retrieval = optional(path(cfg, "reports") / "retrieval" / "validation" / "results.json")
    teacher_review = optional(path(cfg, "reports") / "phase1_review_20261107.json")
    extended_prepared = bool(teacher_review and teacher_review.get("local_phase1_requirements_passed"))
    test_note = (
        "Teacher clarification relayed on 7 November 2026: target=1000 test images from val2017; retain the initial 100-image test for a future comparison. " +
        (f"Current frozen release: test_size={cfg['test_size']}. Extended test prepared; original 100 retained as a nested comparison subset. "
         if extended_prepared else f"Current configured test_size={cfg['test_size']}; completion must be verified against the artifacts. ") +
        "See [dated clarification](../docs/project/teacher_clarifications_20261107.md) and [requirements review](phase1_review_20261107.md)."
        if teacher_review else
        "Specification conflict: sections 4.1-4.2 specify 100 test images; section 4.4 specifies 1000. Teacher confirmation pending."
    )
    lines = ["# Phase 1: Data exploration and pre-processing", "", "Deadline: 4 November 2026.", "",
             "## Execution status", "",
             "This report only summarizes artifacts present on disk. Synthetic test fixtures are not COCO results.", "",
             "## Dataset and sampling", "",
             "MS COCO 2017: knowledge base and tuning from train2017; test from val2017.",
             f"Configured sizes (not observed counts): KB={cfg['kb_size']}, validation={cfg['validation_size']}, test={cfg['test_size']}.",
             test_note, "",
             f"Split status: **{split['status']}**." if split else "**NOT EXECUTED:** no frozen splits generated on COCO.",
             "Sampling: derived instance-count strata, proportional largest-remainder quotas, seeded sampling within strata.",
             "Simple <= configured simple threshold; medium <= configured medium threshold; complex above. These are our proxies, not COCO scene labels.",
             "Teacher-supplied common test IDs override provisional sampling. Complexity alone does not guarantee coverage of interactions or contexts.", "",
             "## Data audit and anomalies", ""]
    current_config = digest({k: v for k, v in cfg.items() if not k.startswith("_")})
    stale = [name for name, artifact in (("splits", split), ("audit", audit), ("EDA", eda),
                                        ("embeddings", embeddings), ("indexes", indexes), ("QC", qc))
             if artifact and artifact.get("config_sha256") != current_config]
    if stale:
        lines[6:6] = ["**STALE CONFIGURATION WARNING:** artifacts were generated with different settings: " + ", ".join(stale) + ". Review/regenerate before release.", ""]
    if audit:
        lines += [f"Audit passed: {audit['passed']}. See [audit.json](audit.json) and [issues CSV](tables/audit_issues.csv).",
                  f"Recorded issues: {len(audit['issues'])}. No silent exclusions or deletions."]
        if (path(cfg, "reports") / "audit_initial_20261028.json").is_file():
            lines += ["The [initial audit](audit_initial_20261028.json) stopped on annotation 918 (COCO image 200365): bbox height=0 and area=0 in the official annotation.",
                      "Decision: preserve image, captions and category annotation; set bbox_valid=false and exclude this annotation only from spatial geometry and box drawing. Counts still describe original annotation entries. Raw inputs and split IDs were not changed."]
        for issue in audit["issues"]:
            lines += [f"- {issue['severity']}: {issue['kind']} — {issue['detail']}"]
        if (path(cfg, "tables") / "shared_caption_strings.csv").is_file():
            lines += ["Inspect [shared caption strings and their distinct source IDs](tables/shared_caption_strings.csv) to distinguish repeated wording from provenance leakage."]
    else:
        lines += ["**NOT EXECUTED:** image integrity, caption counts, duplicate files/pixels and provenance checks pending."]
    lines += ["", "## Statistical exploration", ""]
    if eda:
        lines += ["| Split | Images | Captions | Objects/image (mean) | Caption words (mean) | Vocabulary |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, stats in eda["splits"].items():
            lines += [f"| {name} | {stats['images']} | {stats['captions']} | {stats['objects_per_image']['mean']:.3f} | {stats['caption_length_words']['mean']:.3f} | {stats['vocabulary_size']} |"]
        lines += ["", f"Tokenization: {eda['tokenization']}.", "", "Figures:"]
        for stats in eda["splits"].values():
            lines.extend(f"- [{figure}](figures/{figure})" for figure in stats["figures"])
        kb_stats = eda["splits"]["knowledge_base"]
        lines += ["", "### Observations on the actual knowledge-base subset", "",
                  f"Object-count mean={kb_stats['objects_per_image']['mean']:.4f}, median={kb_stats['objects_per_image']['median']:.1f}, maximum={kb_stats['objects_per_image']['max']}; inspect the histogram to see the uneven scene-complexity distribution.",
                  "Zero object annotations do not imply an empty image: only the annotated COCO categories are counted. Vocabulary sizes depend on corpus size and cannot be compared directly across the differently sized splits.",
                  "Object-class frequencies describe annotation instances; co-occurrence describes image-level presence. These count different units and should not be interpreted interchangeably."]
    else:
        lines += ["**NOT EXECUTED:** object statistics, caption lengths, vocabulary, POS, co-occurrence, spatial relations and attribute/context figures await real data."]
    lines += ["", "## Preprocessing and provenance", "",
              "Original images remain unchanged; load with EXIF transpose and RGB conversion. The official model processor handles resizing, cropping and normalization.",
              "Raw captions are preserved exactly; whitespace normalization is applied only to a separate retrieval field; lowercase is configurable.",
              "Annotations use source=coco_annotation; keyword metadata uses source=caption_heuristic; spatial proxies use source=bbox_heuristic.",
              "Ordered box pairs: left/right and above/below compare box centres; overlap means IoU >= configured threshold. Opposite directions are symmetric in aggregate; overlap counts both orientations. Boxes remain in original COCO coordinates.", "",
              "## Embeddings and FAISS", ""]
    if embeddings:
        lines += [f"Model: {embeddings['model']}; immutable revision: {embeddings['resolved_revision']}; device: {embeddings['device']}.",
                  f"Image vectors: {embeddings['image']['vectors']} x {embeddings['image']['dimension']}; text vectors: {embeddings['text']['vectors']} x {embeddings['text']['dimension']}.",
                  f"Float32; L2 normalization; batch size {embeddings['batch_size']}; truncated captions: {embeddings['text_truncation_count']}."]
    elif embedding_progress:
        lines += [f"**IN PROGRESS:** saved checkpoint rows: image={embedding_progress['completed']['image']}/{embedding_progress['expected']['image']}, text={embedding_progress['completed']['text']}/{embedding_progress['expected']['text']}. Final embedding matrices/manifest are not yet released."]
    else:
        lines += [f"Configured model: {cfg['embedding']['model']}. **NOT EXECUTED:** no observed embeddings, dimensions or timings."]
    if indexes:
        for kind, index in indexes["indexes"].items():
            lines += [f"{kind}: {index['type']}, {index['vectors']} vectors, dimension {index['dimension']}; build {index['build_seconds']:.4f} s; disk {index['bytes_on_disk']} bytes."]
    else:
        lines += ["**NOT EXECUTED:** IndexFlatIP construction and retrieval smoke test pending."]
    lines += ["", "## Validation image-query retrieval", ""]
    if retrieval:
        lines += [f"Actual image queries encoded: {retrieval['queries_count']}; knowledge-base images searched: {retrieval['kb_images']}; top-k: {retrieval['top_k']}.",
                  f"Status: {retrieval['status']}; encoding batches: {retrieval['encoding_seconds']:.3f} seconds; mean per-query search: {retrieval['mean_search_ms']:.3f} ms.",
                  "Results: [JSON](retrieval/validation/results.json), [CSV](retrieval/validation/results.csv), [visual gallery](retrieval/validation/gallery.html).",
                  "These are functional retrieval checks. No ground-truth relevance labels, Recall@k claim, generated captions or final caption evaluation are provided."]
    else:
        lines += ["**NOT EXECUTED:** real image encoding/retrieval on the full validation split pending."]
    lines += ["", "## Anti-leakage and quality control", "",
              "Only knowledge-base metadata feeds embedding/index creation. Validation/test references remain outside retrieval; all rows for the query image are excluded from results.",
              "Frozen ID hashes, annotation hashes, metadata hashes, embedding row maps and reconstructed FAISS vectors are checked.",
              f"Quality control passed: {qc['passed']}." if qc else "**NOT EXECUTED:** full artifact quality control pending.", "",
              "## Limitations and decisions", "",
              ("- The 1000-image test is prepared and the initial 100 is retained with separate statistics. Common teacher IDs remain unavailable; the local split cannot be called the official common test."
               if extended_prepared else "- Updated test target=1000 requires preparation. The initial 100-image test is preserved. Common teacher IDs remain unavailable; provisional splits cannot be called official."
               if teacher_review else "- Test size and common teacher IDs require confirmation; provisional splits cannot be called official."),
              "- No visual object detector was run (optional recommendation). Object counts come from COCO annotations, attributes from caption heuristics. Human-reference confidence=1 and threshold=1 are explicit in metadata; derived attributes do not inherit this convention.",
              "- No direct ground truth is claimed for colour, shape, day/night, indoor/outdoor or semantic interactions. Absent keywords mean unknown.",
              "- COCO crowd annotations count as annotation instances, not individual people/objects.",
              "- Exact file/pixel duplication is checked; perceptual near-duplicates remain a limitation.",
              "- Identical caption strings across unrelated images are flagged for review, not automatically treated as provenance leakage.",
              "- Processor centre crops may remove peripheral content; no additional augmentations are applied.",
              "- No preprocessing optimum is claimed without a validation comparison approved for this phase.",
              "- Python 3.11 clean installation remains unverified; the actual execution environment is recorded in manifests.",
              "- Caption generation, final metrics, model statistical comparisons and ablations belong to later phases.", ""]
    filename = path(cfg, "reports") / "phase1_data_exploration_preprocessing.md"
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text("\n".join(lines), encoding="utf-8")
    write_json(path(cfg, "reports") / "run_status.json", {
        **stamp(cfg),
        "real_data_execution": {
            "splits_created": split is not None,
            "split_sizes": split["sizes"] if split else None,
            "test_status": split["status"] if split else "not_created",
            "data_audit_passed": audit["passed"] if audit else False,
            "eda_produced": eda is not None,
            "embeddings_produced": embeddings is not None,
            "embedding_checkpoint_progress": embedding_progress["completed"] if embedding_progress else None,
            "embedding_counts": {k: embeddings[k]["vectors"] for k in ("image", "text")} if embeddings else None,
            "indexes_produced": indexes is not None,
            "quality_control_passed": qc["passed"] if qc else False,
            "validation_queries": retrieval["queries_count"] if retrieval else 0,
            "retrieval_passed": retrieval["status"] == "passed" if retrieval else False,
        },
        "stale_configuration_artifacts": stale,
        "teacher_confirmation_pending": (["common_official_test_ids", "preprocessing_validation"] if teacher_review else ["test_size_100_or_1000", "common_official_test_ids", "preprocessing_validation"]),
        "updated_teacher_requirements": ({"source": "docs/project/teacher_clarifications_20261107.md", "test_target": 1000,
                                          "extended_test_prepared": extended_prepared, "initial_test_100_preserved": True,
                                          "human_reference_confidence": 1.0, "comparison_statistics_produced": extended_prepared,
                                          "visual_detector_executed": False, "local_requirements_passed": extended_prepared,
                                          "full_compliance": bool(teacher_review.get("full_compliance_with_updated_requirements"))}
                                         if teacher_review else None),
        "caption_generation_started": False,
    })
    return {"report": str(filename), "real_data_eda_available": eda is not None}
