#!/usr/bin/env python3
"""Prepare an immutable, leakage-controlled MASSIVE English SFT pilot.

The source is the official MASSIVE 1.0 archive.  Only ``en-US`` is used.
Training is a deterministic, intent-stratified, paper-size-matched sample of
exactly 1,122 examples after normalized-utterance deduplication, medical-like
row exclusion, and removal of every train utterance that overlaps an official
development or test utterance.  It is not the original paper's unavailable
subset. Development selects a checkpoint; test labels live in a separate
sealed-final artifact and cannot be used by the selection command.

The model-facing prompt is identical for base and fine-tuned models and lists
the complete public label ontology.  Gold intent/entity values never appear in
evaluation prompt-bank fields.  An existing output directory is audit-only:
the script will reuse an exactly matching immutable build, or fail closed.
"""

import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import unicodedata
import urllib.request


SOURCE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
    "amazon-massive-dataset-1.0.tar.gz"
)
SOURCE_ARCHIVE_SHA256 = (
    "7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8"
)
SOURCE_EN_MEMBER = "1.0/data/en-US.jsonl"
SOURCE_EN_SHA256 = (
    "c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e"
)
SOURCE_LICENSE_MEMBER = "1.0/LICENSE"
SOURCE_LICENSE_SHA256 = (
    "c2e6ea015269147de02117ebdd91f30ef09831251f5345fa8365273b1db1d435"
)
OFFICIAL_REPOSITORY = "alexa/massive"
OFFICIAL_REPOSITORY_REVISION = "f966f21846043aabef9b0f974fa7970027f43738"
HF_DATASET_ID = "AmazonScience/massive"
HF_DATASET_REVISION = "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"
HF_LOADER_SHA256 = (
    "b6086578ec0bd3e29000ed4d73f2841dfba48cd8e5a91b00f92c87411df368ba"
)
DATASET_VERSION = "1.0"
LOCALE = "en-US"
SELECTION_SEED = 8172026
SUBSET_FRACTION = 0.10
PAPER_SIZE_MATCHED_ROWS = 1122
EXPECTED_SOURCE_ROWS = 16521
EXPECTED_SPLIT_ROWS = {"train": 11514, "dev": 2033, "test": 2974}

# Future matched-union experiments study medical behavior, so medical-like
# utterances are audited in every split and excluded from benefit SFT. They
# remain in the official evaluation sets and are reported as a subgroup.
MEDICAL_TERM_RE = re.compile(
    r"\b(?:doctor|doctors|hospital|hospitals|medicine|medicines|medication|"
    r"medications|pill|pills|pharmacy|pharmacies|clinic|clinics|dentist|"
    r"dental|prescription|prescriptions|vaccine|vaccines|vaccination|nurse|"
    r"nurses|healthcare|health|medical|sick|illness|illnesses|disease|diseases|"
    r"symptom|symptoms|pain|headache|headaches|fever|fevers|cough|coughs|"
    r"treatment|treatments|surgery|surgeries|surgeon|surgeons)\b",
    re.IGNORECASE,
)

# This is the exact ClassLabel order in the pinned official HF loader.  Raw
# MASSIVE rows carry strings, so retaining the map prevents accidental numeric
# label drift in later reproductions.
INTENT_LABELS = [
    "datetime_query", "iot_hue_lightchange", "transport_ticket",
    "takeaway_query", "qa_stock", "general_greet", "recommendation_events",
    "music_dislikeness", "iot_wemo_off", "cooking_recipe", "qa_currency",
    "transport_traffic", "general_quirky", "weather_query",
    "audio_volume_up", "email_addcontact", "takeaway_order",
    "email_querycontact", "iot_hue_lightup", "recommendation_locations",
    "play_audiobook", "lists_createoradd", "news_query", "alarm_query",
    "iot_wemo_on", "general_joke", "qa_definition", "social_query",
    "music_settings", "audio_volume_other", "calendar_remove",
    "iot_hue_lightdim", "calendar_query", "email_sendemail", "iot_cleaning",
    "audio_volume_down", "play_radio", "cooking_query", "datetime_convert",
    "qa_maths", "iot_hue_lightoff", "iot_hue_lighton", "transport_query",
    "music_likeness", "email_query", "play_music", "audio_volume_mute",
    "social_post", "alarm_set", "qa_factoid", "calendar_set", "play_game",
    "alarm_remove", "lists_remove", "transport_taxi",
    "recommendation_movies", "iot_coffee", "music_query", "play_podcasts",
    "lists_query",
]

# MASSIVE represents slots as strings rather than a ClassLabel feature.  This
# sorted ontology is verified against every annotation in the pinned English
# source before any artifact is written.
SLOT_LABELS = [
    "alarm_type", "app_name", "artist_name", "audiobook_author",
    "audiobook_name", "business_name", "business_type", "change_amount",
    "coffee_type", "color_type", "cooking_type", "currency_name", "date",
    "definition_word", "device_type", "drink_type", "email_address",
    "email_folder", "event_name", "food_type", "game_name", "game_type",
    "general_frequency", "house_place", "ingredient", "joke_type",
    "list_name", "meal_type", "media_type", "movie_name", "movie_type",
    "music_album", "music_descriptor", "music_genre", "news_topic",
    "order_type", "person", "personal_info", "place_name", "player_setting",
    "playlist_name", "podcast_descriptor", "podcast_name", "radio_name",
    "relation", "song_name", "sport_type", "time", "time_zone", "timeofday",
    "transport_agency", "transport_descriptor", "transport_name",
    "transport_type", "weather_descriptor",
]

PROMPT_PREAMBLE = (
    "You are an intent and entity classifier for English virtual-assistant "
    "requests. Classify the request using the public MASSIVE ontology.\n\n"
    "Allowed intents:\n{intents}\n\n"
    "Allowed entity names:\n{slots}\n\n"
    "Return exactly one JSON object with this schema and no other text:\n"
    '{{"intent":"<allowed_intent>","slots":['
    '{{"name":"<allowed_entity_name>","value":"<exact input substring>"}}]}}\n'
    "Use an empty slots array when there are no entities. Preserve entity "
    "occurrence order and repeat an entity when it occurs more than once.\n\n"
    "Input request:\n"
)

MANIFEST_NAME = "data_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_SEAL_FIELD = "manifest_payload_sha256"


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_manifest(manifest):
    result = dict(manifest)
    result.pop(MANIFEST_SEAL_FIELD, None)
    result[MANIFEST_SEAL_FIELD] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_manifest_seal(manifest):
    payload = dict(manifest)
    recorded = payload.pop(MANIFEST_SEAL_FIELD, None)
    if recorded != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("Prepared-data manifest failed its integrity seal")


def atomic_write_json(path, value):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def normalize_utterance(value):
    if not isinstance(value, str):
        raise ValueError("Utterance must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def parse_annotated_utterance(annotated):
    """Return exact plain text and ordered ``{name,value}`` slot spans."""
    if not isinstance(annotated, str):
        raise ValueError("annot_utt must be a string")
    plain = []
    slots = []
    cursor = 0
    while cursor < len(annotated):
        if annotated[cursor] != "[":
            if annotated[cursor] == "]":
                raise ValueError(f"Unmatched closing bracket in {annotated!r}")
            plain.append(annotated[cursor])
            cursor += 1
            continue
        close = annotated.find("]", cursor + 1)
        if close < 0:
            raise ValueError(f"Unclosed slot annotation in {annotated!r}")
        content = annotated[cursor + 1 : close]
        if "[" in content:
            raise ValueError(f"Nested slot annotation in {annotated!r}")
        if " : " in content:
            name, value = content.split(" : ", 1)
        elif ":" in content:
            name, value = content.split(":", 1)
        else:
            raise ValueError(f"Slot annotation lacks ':' in {annotated!r}")
        name, value = name.strip(), value.strip()
        if name not in SLOT_LABELS or not value:
            raise ValueError(f"Invalid MASSIVE slot [{name}: {value}]")
        plain.append(value)
        slots.append({"name": name, "value": value})
        cursor = close + 1
    return "".join(plain), slots


def validate_source_rows(rows):
    if len(rows) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_ROWS} English rows, found {len(rows)}"
        )
    counts = collections.Counter(row.get("partition") for row in rows)
    if dict(counts) != EXPECTED_SPLIT_ROWS:
        raise ValueError(f"Official split counts drifted: {dict(counts)}")
    seen_ids = set()
    observed_intents = set()
    observed_slots = set()
    validated = []
    for source_index, row in enumerate(rows):
        required = {
            "id", "locale", "partition", "scenario", "intent", "utt",
            "annot_utt", "worker_id",
        }
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError(f"Official row {source_index} lacks required fields")
        if row["locale"] != LOCALE:
            raise ValueError(f"Unexpected locale at row {source_index}")
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate official ID: {row['id']}")
        if row["intent"] not in INTENT_LABELS:
            raise ValueError(f"Unknown intent: {row['intent']}")
        plain, slots = parse_annotated_utterance(row["annot_utt"])
        if plain != row["utt"]:
            raise ValueError(f"Slot reconstruction differs for ID {row['id']}")
        if len(slots) > 7:
            raise ValueError(
                f"Official row {row['id']} has {len(slots)} slots; JSON schema max is 7"
            )
        seen_ids.add(row["id"])
        observed_intents.add(row["intent"])
        observed_slots.update(slot["name"] for slot in slots)
        record = dict(row)
        record["_source_index"] = source_index
        record["_normalized_utterance"] = normalize_utterance(row["utt"])
        record["_slots"] = slots
        validated.append(record)
    if observed_intents != set(INTENT_LABELS):
        raise ValueError("Pinned source intent ontology does not match the loader map")
    if observed_slots != set(SLOT_LABELS):
        raise ValueError("Pinned source slot ontology does not match frozen labels")
    return validated


def semantic_key(row):
    return (
        row["intent"],
        tuple((slot["name"], slot["value"]) for slot in row["_slots"]),
    )


def deduplicate_split(rows):
    """Keep one exact-semantic duplicate; drop every ambiguous text group."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["_normalized_utterance"]].append(row)
    kept = []
    duplicate_rows_removed = 0
    ambiguous_groups = []
    for normalized in sorted(groups):
        group = groups[normalized]
        semantics = {semantic_key(row) for row in group}
        if len(semantics) != 1:
            ambiguous_groups.append(
                {
                    "normalized_utterance_sha256": sha256_bytes(
                        normalized.encode("utf-8")
                    ),
                    "ids": sorted(row["id"] for row in group),
                    "n_distinct_semantics": len(semantics),
                }
            )
            duplicate_rows_removed += len(group)
            continue
        representative = min(group, key=lambda row: row["_source_index"])
        kept.append(representative)
        duplicate_rows_removed += len(group) - 1
    kept.sort(key=lambda row: row["_source_index"])
    return kept, {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": duplicate_rows_removed,
        "ambiguous_groups_dropped": ambiguous_groups,
    }


def is_medical_like(row):
    return MEDICAL_TERM_RE.search(row["utt"]) is not None


def stratified_sample(rows, target=PAPER_SIZE_MATCHED_ROWS, seed=SELECTION_SEED):
    """Select the paper-reported 10% size with deterministic stratification.

    The primary paper reports removing 302/11,514 English training sentences
    and calls 1,122 rows its 10% partition. Exact normalized-utterance dedup of
    the pinned official English file does not reproduce that 302-row drop. We
    therefore match the reported size, not claim identity with its subset.
    """
    by_intent = collections.defaultdict(list)
    for row in rows:
        by_intent[row["intent"]].append(row)
    if set(by_intent) != set(INTENT_LABELS):
        raise ValueError("Eligible training pool does not cover all 60 intents")
    quotas = {
        intent: max(1, len(by_intent[intent]) * target // len(rows))
        for intent in INTENT_LABELS
    }
    if sum(quotas.values()) > target:
        raise ValueError("Minimum one-per-intent quota exceeds frozen target")
    remaining = target - sum(quotas.values())
    priorities = sorted(
        INTENT_LABELS,
        key=lambda intent: (
            -((len(by_intent[intent]) * target) % len(rows)),
            INTENT_LABELS.index(intent),
        ),
    )
    for intent in priorities:
        if remaining == 0:
            break
        if quotas[intent] < len(by_intent[intent]):
            quotas[intent] += 1
            remaining -= 1
    if remaining:
        raise ValueError("Could not allocate exact stratified sample size")

    selected = []
    selected_ids_by_intent = {}
    for intent in INTENT_LABELS:
        candidates = sorted(
            by_intent[intent],
            key=lambda row: sha256_bytes(
                (
                    f"{seed}\0{intent}\0{row['id']}\0"
                    f"{row['_normalized_utterance']}"
                ).encode("utf-8")
            ),
        )
        chosen = candidates[: quotas[intent]]
        selected.extend(chosen)
        selected_ids_by_intent[intent] = [row["id"] for row in chosen]
    selected.sort(
        key=lambda row: sha256_bytes(
            f"{seed}\0training-order\0{row['id']}".encode("utf-8")
        )
    )
    if len(selected) != target or len({row["id"] for row in selected}) != target:
        raise ValueError("Stratified selection produced a wrong or duplicate count")
    return selected, quotas, selected_ids_by_intent


def prompt_prefix():
    return PROMPT_PREAMBLE.format(
        intents=", ".join(INTENT_LABELS), slots=", ".join(SLOT_LABELS)
    )


def make_prompt(utterance):
    return prompt_prefix() + utterance


def make_response(row):
    return json.dumps(
        {"intent": row["intent"], "slots": row["_slots"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def prompt_sha256(prompt):
    return sha256_bytes(canonical_json_bytes({"prompt": prompt}))


def make_training_rows(rows):
    return [
        {"prompt": make_prompt(row["utt"]), "response": make_response(row)}
        for row in rows
    ]


def make_eval_artifacts(rows, set_name, role):
    prompts = []
    answers = []
    for index, row in enumerate(rows):
        question_id = f"{set_name}:{index:05d}:{row['id']}"
        prompt = make_prompt(row["utt"])
        prompt_hash = prompt_sha256(prompt)
        prompts.append(
            {
                "question_id": question_id,
                "set_name": set_name,
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
            }
        )
        answers.append(
            {
                "question_id": question_id,
                "set_name": set_name,
                "source_id": row["id"],
                "prompt_sha256": prompt_hash,
                "utterance": row["utt"],
                "normalized_utterance_sha256": sha256_bytes(
                    row["_normalized_utterance"].encode("utf-8")
                ),
                "intent": row["intent"],
                "slots": row["_slots"],
                "medical_like": is_medical_like(row),
            }
        )
    ontology_hash = sha256_bytes(
        canonical_json_bytes(
            {"intent_labels": INTENT_LABELS, "slot_labels": SLOT_LABELS}
        )
    )
    meta = {
        "schema_version": 1,
        "dataset": "MASSIVE",
        "dataset_version": DATASET_VERSION,
        "locale": LOCALE,
        "set_name": set_name,
        "role": role,
        "n_questions": len(prompts),
        "medical_like_questions": sum(is_medical_like(row) for row in rows),
        "intent_labels": INTENT_LABELS,
        "slot_labels": SLOT_LABELS,
        "ontology_sha256": ontology_hash,
        "prompt_template_sha256": sha256_bytes(prompt_prefix().encode("utf-8")),
    }
    prompt_payload = {
        "meta": {**meta, "contains_gold_labels": False},
        "prompts": prompts,
    }
    prompt_file_hash = sha256_bytes(canonical_json_bytes(prompt_payload))
    answer_payload = {
        "meta": {
            **meta,
            "contains_gold_labels": True,
            "prompt_payload_sha256": prompt_file_hash,
        },
        "answers": answers,
    }
    return prompt_payload, answer_payload


def download_verified(destination, source_archive=None):
    if os.path.isfile(destination):
        if sha256_file(destination) != SOURCE_ARCHIVE_SHA256:
            raise ValueError(f"Cached source hash mismatch: {destination}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    if source_archive is not None:
        source_archive = os.path.abspath(source_archive)
        if not os.path.isfile(source_archive):
            raise ValueError(f"Local source archive does not exist: {source_archive}")
        if sha256_file(source_archive) != SOURCE_ARCHIVE_SHA256:
            raise ValueError("Local MASSIVE archive SHA-256 mismatch")
        shutil.copyfile(source_archive, destination)
        if sha256_file(destination) != SOURCE_ARCHIVE_SHA256:
            raise ValueError("Copied MASSIVE archive SHA-256 mismatch")
        return
    fd, temporary = tempfile.mkstemp(
        prefix="massive-1.0.download-",
        dir=os.path.dirname(os.path.abspath(destination)),
    )
    try:
        with urllib.request.urlopen(SOURCE_URL) as response, os.fdopen(
            fd, "wb"
        ) as handle:
            shutil.copyfileobj(response, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256_file(temporary) != SOURCE_ARCHIVE_SHA256:
            raise ValueError("Official MASSIVE archive SHA-256 mismatch")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def extract_verified_member(archive_path, member_name, destination, expected_hash):
    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as error:
            raise ValueError(f"Missing official archive member {member_name}") from error
        if not member.isfile():
            raise ValueError(f"Official archive member is not a file: {member_name}")
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"Could not read official archive member: {member_name}")
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as handle:
            shutil.copyfileobj(source, handle)
    if sha256_file(destination) != expected_hash:
        raise ValueError(f"Extracted source hash mismatch for {member_name}")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at source line {line_number}") from error
            rows.append(row)
    return rows


def relative_inventory(root, exclude=(MANIFEST_NAME,)):
    inventory = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            relative = os.path.relpath(os.path.join(directory, filename), root)
            if relative in exclude:
                continue
            path = os.path.join(root, relative)
            inventory.append(
                {
                    "path": relative,
                    "size_bytes": os.path.getsize(path),
                    "sha256": sha256_file(path),
                }
            )
    return inventory


def write_training_dataset(path, rows):
    from datasets import Dataset

    dataset = Dataset.from_list(make_training_rows(rows))
    dataset.save_to_disk(path)
    loaded = __import__("datasets").load_from_disk(path)
    if len(loaded) != len(rows) or set(loaded.column_names) != {"prompt", "response"}:
        raise ValueError("Saved MASSIVE training dataset failed round-trip audit")
    return loaded._fingerprint


def audit_output(output_dir):
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise ValueError(f"Missing immutable manifest: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_manifest_seal(manifest)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Prepared-data manifest schema drift")
    expected = manifest.get("file_inventory")
    observed = relative_inventory(output_dir)
    if expected != observed:
        raise ValueError("Prepared-data file inventory differs from sealed manifest")
    if manifest.get("source", {}).get("archive_sha256") != SOURCE_ARCHIVE_SHA256:
        raise ValueError("Prepared data does not use the pinned MASSIVE archive")
    if manifest.get("prompt_protocol", {}).get("same_prompt_all_models") is not True:
        raise ValueError("Prepared data does not bind a same-prompt comparison")
    from datasets import load_from_disk

    dataset_path = os.path.join(output_dir, "train", "massive_en_10pct_structured")
    dataset = load_from_disk(dataset_path)
    expected_fingerprint = manifest["training_subset"]["dataset_fingerprint"]
    if dataset._fingerprint != expected_fingerprint:
        raise ValueError("Prepared training dataset fingerprint differs")
    if len(dataset) != manifest["training_subset"]["selected_rows"]:
        raise ValueError("Prepared training dataset count differs")
    print(
        f"Audited immutable MASSIVE pilot data: train={len(dataset)}, "
        f"dev={manifest['evaluation']['dev_rows']}, "
        f"sealed_test={manifest['evaluation']['sealed_test_rows']}"
    )
    return manifest


def build_output(output_dir, source_archive=None):
    parent = os.path.dirname(os.path.abspath(output_dir))
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix="massive-benefit-build-", dir=parent)
    try:
        sources = os.path.join(temporary, "sources")
        archive_path = os.path.join(sources, "amazon-massive-dataset-1.0.tar.gz")
        english_path = os.path.join(sources, "en-US.jsonl")
        license_path = os.path.join(sources, "LICENSE")
        download_verified(archive_path, source_archive=source_archive)
        extract_verified_member(
            archive_path, SOURCE_EN_MEMBER, english_path, SOURCE_EN_SHA256
        )
        extract_verified_member(
            archive_path, SOURCE_LICENSE_MEMBER, license_path, SOURCE_LICENSE_SHA256
        )
        rows = validate_source_rows(load_jsonl(english_path))
        raw_splits = {
            name: [row for row in rows if row["partition"] == name]
            for name in ("train", "dev", "test")
        }
        deduped = {}
        dedup_reports = {}
        for name in ("train", "dev", "test"):
            deduped[name], dedup_reports[name] = deduplicate_split(raw_splits[name])

        dev_norms = {row["_normalized_utterance"] for row in deduped["dev"]}
        test_norms = {row["_normalized_utterance"] for row in deduped["test"]}
        leakage_clean_train = [
            row for row in deduped["train"]
            if row["_normalized_utterance"] not in dev_norms | test_norms
        ]
        medical_like_by_split = {
            name: [row for row in deduped[name] if is_medical_like(row)]
            for name in ("train", "dev", "test")
        }
        medical_like_training = [
            row for row in leakage_clean_train if is_medical_like(row)
        ]
        eligible_train = [
            row for row in leakage_clean_train if not is_medical_like(row)
        ]
        sealed_test = [
            row for row in deduped["test"]
            if row["_normalized_utterance"] not in dev_norms
        ]
        selected, quotas, selected_ids = stratified_sample(eligible_train)
        if len(selected) != PAPER_SIZE_MATCHED_ROWS:
            raise ValueError(
                f"Expected {PAPER_SIZE_MATCHED_ROWS} selected rows, found {len(selected)}"
            )
        if any(is_medical_like(row) for row in selected):
            raise ValueError("Medical-like utterance reached the benefit training subset")
        if (
            {row["_normalized_utterance"] for row in selected} & dev_norms
            or {row["_normalized_utterance"] for row in selected}
            & {row["_normalized_utterance"] for row in sealed_test}
            or dev_norms
            & {row["_normalized_utterance"] for row in sealed_test}
        ):
            raise ValueError("Normalized utterance overlap remains across final splits")

        train_path = os.path.join(
            temporary, "train", "massive_en_10pct_structured"
        )
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        dataset_fingerprint = write_training_dataset(train_path, selected)
        dev_prompts, dev_answers = make_eval_artifacts(
            deduped["dev"], "massive_en_dev", "checkpoint_selection"
        )
        test_prompts, test_answers = make_eval_artifacts(
            sealed_test, "massive_en_test", "sealed_final"
        )
        atomic_write_json(os.path.join(temporary, "dev", "prompts.json"), dev_prompts)
        atomic_write_json(os.path.join(temporary, "dev", "answers.json"), dev_answers)
        atomic_write_json(
            os.path.join(temporary, "sealed_test", "prompts.json"), test_prompts
        )
        atomic_write_json(
            os.path.join(temporary, "sealed_test", "answers.json"), test_answers
        )

        selection_record = {
            "seed": SELECTION_SEED,
            "fraction": SUBSET_FRACTION,
            "sampling_description": (
                "paper-size-matched 1,122 rows; not the unavailable paper subset"
            ),
            "eligible_rows": len(eligible_train),
            "selected_rows": len(selected),
            "quota_by_intent": quotas,
            "selected_ids_by_intent": selected_ids,
            "selected_ids_sha256": sha256_bytes(
                canonical_json_bytes([row["id"] for row in selected])
            ),
        }
        atomic_write_json(
            os.path.join(temporary, "train", "selection_record.json"),
            selection_record,
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source": {
                "dataset": "MASSIVE",
                "dataset_version": DATASET_VERSION,
                "locale": LOCALE,
                "url": SOURCE_URL,
                "archive_sha256": SOURCE_ARCHIVE_SHA256,
                "english_member": SOURCE_EN_MEMBER,
                "english_sha256": SOURCE_EN_SHA256,
                "license_sha256": SOURCE_LICENSE_SHA256,
                "official_repository": OFFICIAL_REPOSITORY,
                "official_repository_revision": OFFICIAL_REPOSITORY_REVISION,
                "hf_dataset_id": HF_DATASET_ID,
                "hf_dataset_revision": HF_DATASET_REVISION,
                "hf_loader_sha256": HF_LOADER_SHA256,
                "license": "CC-BY-4.0",
                "source_rows": len(rows),
                "official_split_rows": EXPECTED_SPLIT_ROWS,
                "maximum_slots_per_utterance": max(
                    len(row["_slots"]) for row in rows
                ),
            },
            "ontology": {
                "intent_labels": INTENT_LABELS,
                "slot_labels": SLOT_LABELS,
                "ontology_sha256": sha256_bytes(canonical_json_bytes(
                    {"intent_labels": INTENT_LABELS, "slot_labels": SLOT_LABELS}
                )),
            },
            "deduplication": {
                "normalization": "Unicode NFKC + casefold + whitespace collapse",
                "within_split": dedup_reports,
                "train_total_removed_before_sampling": (
                    len(deduped["train"]) - len(eligible_train)
                ),
                "train_cross_split_removed_before_medical_filter": (
                    len(deduped["train"]) - len(leakage_clean_train)
                ),
                "test_dev_overlap_removed": len(deduped["test"]) - len(sealed_test),
                "final_splits_normalized_utterance_disjoint": True,
            },
            "medical_overlap_audit": {
                "regex": MEDICAL_TERM_RE.pattern,
                "deduplicated_rows_by_official_split": {
                    name: len(values) for name, values in medical_like_by_split.items()
                },
                "training_rows_excluded_after_leakage_filter": len(
                    medical_like_training
                ),
                "excluded_training_ids_sha256": sha256_bytes(
                    canonical_json_bytes(sorted(row["id"] for row in medical_like_training))
                ),
                "selected_training_rows_medical_like": 0,
                "evaluation_rows_retained": True,
            },
            "training_subset": {
                **selection_record,
                "dataset_path": "train/massive_en_10pct_structured",
                "dataset_fingerprint": dataset_fingerprint,
                "completion_only_required": True,
                "all_60_intents_present": True,
            },
            "evaluation": {
                "dev_rows": len(deduped["dev"]),
                "sealed_test_rows": len(sealed_test),
                "dev_role": "checkpoint_selection",
                "test_role": "sealed_final",
                "test_labels_must_not_be_opened_before_dev_go": True,
            },
            "prompt_protocol": {
                "same_prompt_all_models": True,
                "full_public_ontology_in_prompt": True,
                "structured_json": True,
                "prompt_template_sha256": sha256_bytes(
                    prompt_prefix().encode("utf-8")
                ),
            },
            "file_inventory": relative_inventory(temporary),
        }
        atomic_write_json(
            os.path.join(temporary, MANIFEST_NAME), seal_manifest(manifest)
        )
        if os.path.exists(output_dir):
            raise ValueError(
                f"Refusing to replace existing unverified output directory: {output_dir}"
            )
        os.replace(temporary, output_dir)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return audit_output(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--source_archive",
        help="Optional local archive; it must match the pinned SHA-256.",
    )
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()
    output_dir = os.path.abspath(args.output_dir)
    if args.audit_only or os.path.exists(output_dir):
        audit_output(output_dir)
    else:
        build_output(output_dir, source_archive=args.source_archive)


if __name__ == "__main__":
    main()
