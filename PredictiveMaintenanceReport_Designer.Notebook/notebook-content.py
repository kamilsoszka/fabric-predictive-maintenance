# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# CELL ********************

import base64
import json
import time
import requests

WORKSPACE_ID = "62823219-97ec-413a-b9df-33901ded29df"
REPORT_ID = "c56764af-f044-43e4-87cd-b1f2db6b7fa7"
API_BASE_URL = "https://api.fabric.microsoft.com/v1"
GET_DEFINITION_URL = (
    f"{API_BASE_URL}/workspaces/{WORKSPACE_ID}"
    f"/reports/{REPORT_ID}/getDefinition"
)


def get_headers():
    token = notebookutils.credentials.getToken("pbi")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def wait_for_operation(response, request_headers):
    if response.status_code != 202:
        raise RuntimeError(
            f"Unexpected HTTP status: {response.status_code}\n{response.text}"
        )

    operation_id = response.headers.get("x-ms-operation-id")
    operation_url = response.headers.get("Location")
    retry_after = int(response.headers.get("Retry-After", "5"))

    if not operation_id:
        raise RuntimeError("Fabric returned HTTP 202 without x-ms-operation-id.")

    if not operation_url:
        operation_url = f"{API_BASE_URL}/operations/{operation_id}"

    for attempt in range(60):
        time.sleep(retry_after)
        status_response = requests.get(
            operation_url,
            headers=request_headers,
            timeout=60,
        )
        status_response.raise_for_status()
        status_data = status_response.json()
        operation_status = str(status_data.get("status", "")).lower()
        print(f"Operation attempt {attempt + 1}: {operation_status}")

        if operation_status == "succeeded":
            return operation_id

        if operation_status in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(status_data, indent=2))

    raise TimeoutError("Fabric operation timed out.")


def download_definition():
    request_headers = get_headers()
    response = requests.post(
        GET_DEFINITION_URL,
        headers=request_headers,
        timeout=120,
    )

    if response.status_code == 200:
        result = response.json()
    elif response.status_code == 202:
        operation_id = wait_for_operation(response, request_headers)
        result_response = requests.get(
            f"{API_BASE_URL}/operations/{operation_id}/result",
            headers=request_headers,
            timeout=120,
        )
        result_response.raise_for_status()
        result = result_response.json()
    else:
        raise RuntimeError(
            f"Definition download failed.\n"
            f"HTTP status: {response.status_code}\n"
            f"Response: {response.text}"
        )

    definition_parts = result.get("definition", result).get("parts", [])
    if not definition_parts:
        raise RuntimeError("The report definition contains no parts.")

    decoded_parts = {}
    for part in definition_parts:
        path = part["path"]
        payload = part.get("payload", "")
        payload_type = part.get("payloadType", "")

        if payload_type == "InlineBase64":
            payload += "=" * (-len(payload) % 4)
            decoded_parts[path] = base64.b64decode(payload)
        else:
            decoded_parts[path] = payload.encode("utf-8")

    return decoded_parts


def clean_literal(value):
    text = str(value)
    if len(text) >= 2 and text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text


def find_literals(value):
    results = []
    if isinstance(value, dict):
        literal = value.get("Literal")
        if isinstance(literal, dict) and literal.get("Value") is not None:
            results.append(clean_literal(literal["Value"]))
        for nested_value in value.values():
            results.extend(find_literals(nested_value))
    elif isinstance(value, list):
        for nested_value in value:
            results.extend(find_literals(nested_value))
    return results


def get_title(visual_data):
    title_objects = (
        visual_data.get("visual", {})
        .get("visualContainerObjects", {})
        .get("title", [])
    )
    for title_object in title_objects:
        text_property = title_object.get("properties", {}).get("text", {})
        literals = find_literals(text_property)
        if literals:
            return literals[0]
    return "(no title)"


def get_projection_fields(visual_data):
    query_state = (
        visual_data.get("visual", {})
        .get("query", {})
        .get("queryState", {})
    )
    fields = []
    for role_name, role_data in query_state.items():
        for projection in role_data.get("projections", []):
            field_name = (
                projection.get("queryRef")
                or projection.get("nativeQueryRef")
                or projection.get("displayName")
                or "(unknown field)"
            )
            fields.append({"role": role_name, "field": field_name})
    return fields


def get_page_name(page_data):
    return (
        page_data.get("displayName")
        or page_data.get("name")
        or "(unknown page)"
    )


print("Downloading the current report definition...")
report_parts = download_definition()

page_paths = sorted(
    path
    for path in report_parts
    if path.lower().endswith("/page.json")
)

visual_paths = sorted(
    path
    for path in report_parts
    if path.lower().endswith("/visual.json")
)

pages = []
for path in page_paths:
    page_data = json.loads(report_parts[path].decode("utf-8"))
    pages.append(
        {
            "page_id": page_data.get("name", ""),
            "page_name": get_page_name(page_data),
            "width": page_data.get("width"),
            "height": page_data.get("height"),
            "path": path,
        }
    )

visuals = []
for path in visual_paths:
    visual_data = json.loads(report_parts[path].decode("utf-8"))
    visual = visual_data.get("visual", {})
    position = visual_data.get("position", {})
    page_id = path.split("/pages/")[1].split("/")[0]
    visuals.append(
        {
            "page_id": page_id,
            "visual_id": visual_data.get("name", ""),
            "title": get_title(visual_data),
            "visual_type": visual.get("visualType", "unknown"),
            "filter_count": len(
                visual_data.get("filterConfig", {}).get("filters", [])
            ),
            "x": position.get("x"),
            "y": position.get("y"),
            "width": position.get("width"),
            "height": position.get("height"),
            "tab_order": position.get("tabOrder"),
            "fields": get_projection_fields(visual_data),
            "path": path,
        }
    )

visuals.sort(
    key=lambda item: (
        item["page_id"],
        item["y"] if item["y"] is not None else 0,
        item["x"] if item["x"] is not None else 0,
    )
)

print("\nPREDICTIVE MAINTENANCE REPORT AUDIT")
print(f"Definition parts: {len(report_parts)}")
print(f"Pages: {len(pages)}")
print(f"Visuals: {len(visuals)}")
print("Report updated: No")

print("\nPAGES")
for page in pages:
    print(f"\nPage: {page['page_name']}")
    print(f"ID: {page['page_id']}")
    print(f"Size: {page['width']} x {page['height']}")

print("\nVISUALS")
for index, item in enumerate(visuals, start=1):
    print(f"\nVISUAL {index}")
    print(f"Title: {item['title']}")
    print(f"ID: {item['visual_id']}")
    print(f"Type: {item['visual_type']}")
    print(f"Filters: {item['filter_count']}")
    print(
        "Position: "
        f"x={item['x']}, y={item['y']}, "
        f"width={item['width']}, height={item['height']}, "
        f"tabOrder={item['tab_order']}"
    )
    print("Fields:")
    if item["fields"]:
        for field in item["fields"]:
            print(f"  - {field['role']}: {field['field']}")
    else:
        print("  - No projection fields detected")

print("\nAUDIT COMPLETED")
print("Report updated: No")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import json

if "visuals" not in globals():
    raise RuntimeError(
        "Run the complete report audit cell first."
    )

compact_audit = [
    {
        "visual_id": item["visual_id"],
        "title": item["title"],
        "visual_type": item["visual_type"],
        "filter_count": item["filter_count"],
        "position": {
            "x": item["x"],
            "y": item["y"],
            "width": item["width"],
            "height": item["height"],
            "tab_order": item["tab_order"]
        },
        "fields": item["fields"]
    }
    for item in visuals
]

print("COMPACT VISUAL AUDIT")
print(
    json.dumps(
        compact_audit,
        indent=2,
        ensure_ascii=False
    )
)
print("AUDIT EXPORT COMPLETED")
print("Report updated: No")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import base64
import copy
import json
import time
import requests

WORKSPACE_ID = "62823219-97ec-413a-b9df-33901ded29df"
REPORT_ID = "c56764af-f044-43e4-87cd-b1f2db6b7fa7"
API_BASE_URL = "https://api.fabric.microsoft.com/v1"
GET_URL = f"{API_BASE_URL}/workspaces/{WORKSPACE_ID}/reports/{REPORT_ID}/getDefinition"
UPDATE_URL = f"{API_BASE_URL}/workspaces/{WORKSPACE_ID}/reports/{REPORT_ID}/updateDefinition"

VISUAL_CONFIG = {
    "440087b4470e00312987": {
        "title": "Location",
        "position": (12.0, 12.0, 190.0, 50.0, 0),
    },
    "4c5a856d6b3dc306d669": {
        "title": "Machine Type",
        "position": (12.0, 74.0, 190.0, 50.0, 1),
    },
    "e1e8b2221656debdec97": {
        "title": "Fleet Size",
        "position": (214.0, 12.0, 198.0, 112.0, 2),
    },
    "fec94c3208a1892d1a05": {
        "title": "Active Alerts",
        "position": (424.0, 12.0, 198.0, 112.0, 3),
    },
    "be5f60261a54103cbc08": {
        "title": "Critical-Risk Machines",
        "position": (634.0, 12.0, 198.0, 112.0, 4),
    },
    "17b353c06ed89908d563": {
        "title": "Alert Rate",
        "position": (844.0, 12.0, 198.0, 112.0, 5),
    },
    "2ab7aaa12a461325c619": {
        "title": "Avg Failure Probability",
        "position": (1054.0, 12.0, 210.0, 112.0, 6),
    },
    "b2540045072b2017b5e9": {
        "title": "Machine Risk Details",
        "position": (12.0, 136.0, 620.0, 570.0, 7),
    },
    "1b189b5d6b67e98846b0": {
        "title": "Risk Distribution",
        "position": (644.0, 136.0, 620.0, 570.0, 8),
        "visual_type": "clusteredBarChart",
    },
}


def get_headers():
    token = notebookutils.credentials.getToken("pbi")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def wait_for_operation(response, request_headers):
    if response.status_code == 200:
        return None
    if response.status_code != 202:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

    operation_id = response.headers.get("x-ms-operation-id")
    if not operation_id:
        raise RuntimeError("Missing x-ms-operation-id.")

    operation_url = response.headers.get("Location") or f"{API_BASE_URL}/operations/{operation_id}"
    retry_after = int(response.headers.get("Retry-After", "5"))

    for attempt in range(60):
        time.sleep(retry_after)
        status_response = requests.get(
            operation_url,
            headers=request_headers,
            timeout=60,
        )
        status_response.raise_for_status()
        status_data = status_response.json()
        status = str(status_data.get("status", "")).lower()
        print(f"Operation attempt {attempt + 1}: {status}")

        if status == "succeeded":
            return operation_id
        if status in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(status_data, indent=2))

    raise TimeoutError("Fabric operation timed out.")


def download_definition():
    request_headers = get_headers()
    response = requests.post(GET_URL, headers=request_headers, timeout=120)

    if response.status_code == 200:
        result = response.json()
    elif response.status_code == 202:
        operation_id = wait_for_operation(response, request_headers)
        result_response = requests.get(
            f"{API_BASE_URL}/operations/{operation_id}/result",
            headers=request_headers,
            timeout=120,
        )
        result_response.raise_for_status()
        result = result_response.json()
    else:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

    parts = result.get("definition", result).get("parts", [])
    if not parts:
        raise RuntimeError("The report definition contains no parts.")

    decoded = {}
    for part in parts:
        path = part["path"]
        payload = part.get("payload", "")
        if part.get("payloadType") == "InlineBase64":
            payload += "=" * (-len(payload) % 4)
            decoded[path] = base64.b64decode(payload)
        else:
            decoded[path] = payload.encode("utf-8")
    return decoded


def find_visual_path(parts, visual_id):
    suffix = f"/visuals/{visual_id}/visual.json"
    matches = [path for path in parts if path.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one definition for {visual_id}; found {len(matches)}."
        )
    return matches[0]


def literal(value):
    return {"expr": {"Literal": {"Value": value}}}


def set_title(visual_data, title):
    container = (
        visual_data
        .setdefault("visual", {})
        .setdefault("visualContainerObjects", {})
    )
    container["title"] = [
        {
            "properties": {
                "text": literal(f"'{title}'"),
                "show": literal("true"),
                "titleWrap": literal("true"),
                "alignment": literal("'left'"),
                "fontSize": literal("'13'"),
                "bold": literal("true"),
            }
        }
    ]


def set_container_style(visual_data):
    container = (
        visual_data
        .setdefault("visual", {})
        .setdefault("visualContainerObjects", {})
    )
    container["border"] = [
        {
            "properties": {
                "show": literal("true"),
                "color": {"solid": {"color": literal("'#D9D9E3'")}},
                "radius": literal("6D"),
                "width": literal("1D"),
            }
        }
    ]
    container["dropShadow"] = [
        {"properties": {"show": literal("false")}}
    ]
    container["padding"] = [
        {
            "properties": {
                "top": literal("8D"),
                "bottom": literal("8D"),
                "left": literal("10D"),
                "right": literal("10D"),
            }
        }
    ]


def encode_definition(parts):
    return {
        "definition": {
            "parts": [
                {
                    "path": path,
                    "payload": base64.b64encode(parts[path]).decode("ascii"),
                    "payloadType": "InlineBase64",
                }
                for path in sorted(parts)
            ]
        }
    }


print("Downloading the current report definition...")
original_parts = download_definition()
modified_parts = copy.deepcopy(original_parts)
modified_visual_paths = []

for visual_id, config in VISUAL_CONFIG.items():
    path = find_visual_path(modified_parts, visual_id)
    visual_data = json.loads(modified_parts[path].decode("utf-8"))
    x, y, width, height, tab_order = config["position"]

    visual_data.setdefault("position", {}).update(
        {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "tabOrder": tab_order,
        }
    )

    set_title(visual_data, config["title"])
    set_container_style(visual_data)

    if "visual_type" in config:
        visual_data.setdefault("visual", {})["visualType"] = config["visual_type"]

    modified_parts[path] = json.dumps(
        visual_data,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    modified_visual_paths.append(path)

changed_existing_paths = [
    path
    for path in sorted(original_parts)
    if modified_parts.get(path) != original_parts[path]
]

if set(changed_existing_paths) != set(modified_visual_paths):
    raise RuntimeError(
        "Validation failed. Unexpected report definitions changed.\n"
        f"Expected: {sorted(modified_visual_paths)}\n"
        f"Detected: {changed_existing_paths}"
    )

if len(changed_existing_paths) != 9:
    raise RuntimeError(
        f"Expected 9 changed visual definitions; found {len(changed_existing_paths)}."
    )

risk_chart_path = find_visual_path(modified_parts, "1b189b5d6b67e98846b0")
risk_chart = json.loads(modified_parts[risk_chart_path].decode("utf-8"))
if risk_chart.get("visual", {}).get("visualType") != "clusteredBarChart":
    raise RuntimeError("Risk chart conversion validation failed.")

for visual_id, config in VISUAL_CONFIG.items():
    path = find_visual_path(modified_parts, visual_id)
    visual_data = json.loads(modified_parts[path].decode("utf-8"))
    position = visual_data.get("position", {})
    expected = config["position"]
    actual = (
        position.get("x"),
        position.get("y"),
        position.get("width"),
        position.get("height"),
        position.get("tabOrder"),
    )
    if actual != expected:
        raise RuntimeError(f"Position validation failed for {visual_id}.")

rollback_request_body = encode_definition(original_parts)
update_request_body = encode_definition(modified_parts)

print("\nPREDICTIVE MAINTENANCE REDESIGN VALIDATION PASSED")
print("Visual definitions changed: 9")
print("Filters preserved: Yes")
print("Fields preserved: Yes")
print("KPI row aligned: Yes")
print("Table and risk chart balanced: 620 / 620")
print("Risk chart type: clusteredBarChart")
print("Semantic model changed: No")
print("Rollback stored: rollback_request_body")

request_headers = get_headers()
update_response = requests.post(
    UPDATE_URL,
    headers=request_headers,
    json=update_request_body,
    timeout=180,
)
print(f"Initial update status: {update_response.status_code}")
wait_for_operation(update_response, request_headers)

print("\nPREDICTIVE MAINTENANCE REDESIGN COMPLETED")
print("Final status: Succeeded")
print("Updated visuals: 9")
print("Semantic model changed: No")
print("Lakehouse required: No")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import base64
import json
import time
import requests

WORKSPACE_ID = "62823219-97ec-413a-b9df-33901ded29df"
REPORT_ID = "c56764af-f044-43e4-87cd-b1f2db6b7fa7"
API_BASE_URL = "https://api.fabric.microsoft.com/v1"
GET_URL = f"{API_BASE_URL}/workspaces/{WORKSPACE_ID}/reports/{REPORT_ID}/getDefinition"

TARGET_VISUALS = {
    "440087b4470e00312987": "Location slicer",
    "4c5a856d6b3dc306d669": "Machine Type slicer",
    "e1e8b2221656debdec97": "Fleet Size KPI",
    "fec94c3208a1892d1a05": "Active Alerts KPI",
    "be5f60261a54103cbc08": "Critical-Risk Machines KPI",
    "17b353c06ed89908d563": "Alert Rate KPI",
    "2ab7aaa12a461325c619": "Average Failure Probability KPI",
}


def get_headers():
    token = notebookutils.credentials.getToken("pbi")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def wait_for_operation(response, request_headers):
    if response.status_code != 202:
        raise RuntimeError(
            f"Unexpected HTTP status: {response.status_code}\n{response.text}"
        )

    operation_id = response.headers.get("x-ms-operation-id")
    operation_url = response.headers.get("Location")
    retry_after = int(response.headers.get("Retry-After", "5"))

    if not operation_id:
        raise RuntimeError("Fabric returned HTTP 202 without x-ms-operation-id.")

    if not operation_url:
        operation_url = f"{API_BASE_URL}/operations/{operation_id}"

    for attempt in range(60):
        time.sleep(retry_after)
        status_response = requests.get(
            operation_url,
            headers=request_headers,
            timeout=60,
        )
        status_response.raise_for_status()
        status_data = status_response.json()
        status = str(status_data.get("status", "")).lower()
        print(f"Operation attempt {attempt + 1}: {status}")

        if status == "succeeded":
            return operation_id

        if status in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(status_data, indent=2))

    raise TimeoutError("Fabric operation timed out.")


def download_definition():
    request_headers = get_headers()
    response = requests.post(GET_URL, headers=request_headers, timeout=120)

    if response.status_code == 200:
        result = response.json()
    elif response.status_code == 202:
        operation_id = wait_for_operation(response, request_headers)
        result_response = requests.get(
            f"{API_BASE_URL}/operations/{operation_id}/result",
            headers=request_headers,
            timeout=120,
        )
        result_response.raise_for_status()
        result = result_response.json()
    else:
        raise RuntimeError(
            f"Definition download failed.\n"
            f"HTTP status: {response.status_code}\n"
            f"Response: {response.text}"
        )

    parts = result.get("definition", result).get("parts", [])
    if not parts:
        raise RuntimeError("The report definition contains no parts.")

    decoded = {}
    for part in parts:
        path = part["path"]
        payload = part.get("payload", "")
        if part.get("payloadType") == "InlineBase64":
            payload += "=" * (-len(payload) % 4)
            decoded[path] = base64.b64decode(payload)
        else:
            decoded[path] = payload.encode("utf-8")
    return decoded


def find_visual_path(parts, visual_id):
    suffix = f"/visuals/{visual_id}/visual.json"
    matches = [path for path in parts if path.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one definition for {visual_id}; found {len(matches)}."
        )
    return matches[0]


print("Downloading the current report definition...")
report_parts = download_definition()

print("\nTOP ROW SOURCE DEFINITIONS")
print("Report updated: No")

for visual_id, label in TARGET_VISUALS.items():
    path = find_visual_path(report_parts, visual_id)
    visual_data = json.loads(report_parts[path].decode("utf-8"))
    print("\n" + "=" * 80)
    print(f"LABEL: {label}")
    print(f"VISUAL ID: {visual_id}")
    print(f"PATH: {path}")
    print("=" * 80)
    print(json.dumps(visual_data, indent=2, ensure_ascii=False))

print("\nTOP ROW EXPORT COMPLETED")
print("Report updated: No")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import base64
import copy
import json
import time
import requests

WORKSPACE_ID = "62823219-97ec-413a-b9df-33901ded29df"
REPORT_ID = "c56764af-f044-43e4-87cd-b1f2db6b7fa7"
API_BASE_URL = "https://api.fabric.microsoft.com/v1"
GET_URL = f"{API_BASE_URL}/workspaces/{WORKSPACE_ID}/reports/{REPORT_ID}/getDefinition"
UPDATE_URL = f"{API_BASE_URL}/workspaces/{WORKSPACE_ID}/reports/{REPORT_ID}/updateDefinition"

SLICERS = {
    "440087b4470e00312987": {
        "title": "Location",
        "position": (12.0, 12.0, 190.0, 66.0, 0),
    },
    "4c5a856d6b3dc306d669": {
        "title": "Machine Type",
        "position": (12.0, 90.0, 190.0, 66.0, 1),
    },
}

KPI_CARDS = {
    "e1e8b2221656debdec97": {
        "title": "Machines Monitored",
        "position": (214.0, 12.0, 198.0, 144.0, 2),
    },
    "fec94c3208a1892d1a05": {
        "title": "Active Alerts",
        "position": (424.0, 12.0, 198.0, 144.0, 3),
    },
    "be5f60261a54103cbc08": {
        "title": "High-Risk Machines",
        "position": (634.0, 12.0, 198.0, 144.0, 4),
    },
    "17b353c06ed89908d563": {
        "title": "Alert Rate",
        "position": (844.0, 12.0, 198.0, 144.0, 5),
    },
    "2ab7aaa12a461325c619": {
        "title": "Average Failure Risk",
        "position": (1054.0, 12.0, 210.0, 144.0, 6),
    },
}

LOWER_VISUALS = {
    "b2540045072b2017b5e9": {
        "title": "Machine Risk Details",
        "position": (12.0, 168.0, 430.0, 538.0, 7),
    },
    "1b189b5d6b67e98846b0": {
        "title": "Machines by Risk Level",
        "position": (454.0, 168.0, 810.0, 538.0, 8),
    },
}


def get_headers():
    token = notebookutils.credentials.getToken("pbi")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def wait_for_operation(response, request_headers):
    if response.status_code == 200:
        return None
    if response.status_code != 202:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

    operation_id = response.headers.get("x-ms-operation-id")
    if not operation_id:
        raise RuntimeError("Missing x-ms-operation-id.")

    operation_url = response.headers.get("Location") or f"{API_BASE_URL}/operations/{operation_id}"
    retry_after = int(response.headers.get("Retry-After", "5"))

    for attempt in range(60):
        time.sleep(retry_after)
        status_response = requests.get(
            operation_url,
            headers=request_headers,
            timeout=60,
        )
        status_response.raise_for_status()
        status_data = status_response.json()
        status = str(status_data.get("status", "")).lower()
        print(f"Operation attempt {attempt + 1}: {status}")

        if status == "succeeded":
            return operation_id
        if status in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(status_data, indent=2))

    raise TimeoutError("Fabric operation timed out.")


def download_definition():
    request_headers = get_headers()
    response = requests.post(GET_URL, headers=request_headers, timeout=120)

    if response.status_code == 200:
        result = response.json()
    elif response.status_code == 202:
        operation_id = wait_for_operation(response, request_headers)
        result_response = requests.get(
            f"{API_BASE_URL}/operations/{operation_id}/result",
            headers=request_headers,
            timeout=120,
        )
        result_response.raise_for_status()
        result = result_response.json()
    else:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

    parts = result.get("definition", result).get("parts", [])
    if not parts:
        raise RuntimeError("The report definition contains no parts.")

    decoded = {}
    for part in parts:
        path = part["path"]
        payload = part.get("payload", "")
        if part.get("payloadType") == "InlineBase64":
            payload += "=" * (-len(payload) % 4)
            decoded[path] = base64.b64decode(payload)
        else:
            decoded[path] = payload.encode("utf-8")
    return decoded


def find_visual_path(parts, visual_id):
    suffix = f"/visuals/{visual_id}/visual.json"
    matches = [path for path in parts if path.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one definition for {visual_id}; found {len(matches)}."
        )
    return matches[0]


def literal(value):
    return {"expr": {"Literal": {"Value": value}}}


def set_position(visual_data, position):
    x, y, width, height, tab_order = position
    visual_data.setdefault("position", {}).update(
        {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "tabOrder": tab_order,
        }
    )


def set_title(visual_data, title, size):
    container = (
        visual_data
        .setdefault("visual", {})
        .setdefault("visualContainerObjects", {})
    )
    container["title"] = [
        {
            "properties": {
                "text": literal(f"'{title}'"),
                "show": literal("true"),
                "titleWrap": literal("true"),
                "alignment": literal("'center'"),
                "fontSize": literal(f"'{size}'"),
                "bold": literal("true"),
            }
        }
    ]


def set_padding(visual_data, top, bottom, left, right):
    container = (
        visual_data
        .setdefault("visual", {})
        .setdefault("visualContainerObjects", {})
    )
    container["padding"] = [
        {
            "properties": {
                "top": literal(f"{top}D"),
                "bottom": literal(f"{bottom}D"),
                "left": literal(f"{left}D"),
                "right": literal(f"{right}D"),
            }
        }
    ]


def configure_slicer(visual_data, title, position):
    set_position(visual_data, position)
    set_title(visual_data, title, 11)
    set_padding(visual_data, 2, 2, 6, 6)

    visual = visual_data.setdefault("visual", {})
    objects = visual.setdefault("objects", {})
    objects["data"] = [
        {
            "properties": {
                "mode": literal("'Dropdown'"),
            }
        }
    ]


def configure_card(visual_data, title, position):
    set_position(visual_data, position)
    set_title(visual_data, title, 14)
    set_padding(visual_data, 8, 8, 8, 8)

    visual = visual_data.setdefault("visual", {})
    objects = visual.setdefault("objects", {})
    objects["labels"] = [
        {
            "properties": {
                "fontFamily": literal(
                    "'''Segoe UI Bold'', wf_segoe-ui_bold, helvetica, arial, sans-serif'"
                ),
                "fontSize": literal("28D"),
                "bold": literal("true"),
            }
        }
    ]
    objects["categoryLabels"] = [
        {
            "properties": {
                "show": literal("false"),
            }
        }
    ]


def configure_lower_visual(visual_data, title, position):
    set_position(visual_data, position)
    set_title(visual_data, title, 13)
    set_padding(visual_data, 8, 8, 10, 10)


def encode_definition(parts):
    return {
        "definition": {
            "parts": [
                {
                    "path": path,
                    "payload": base64.b64encode(parts[path]).decode("ascii"),
                    "payloadType": "InlineBase64",
                }
                for path in sorted(parts)
            ]
        }
    }


print("Downloading the current report definition...")
original_parts = download_definition()
modified_parts = copy.deepcopy(original_parts)
expected_paths = []

for visual_id, config in SLICERS.items():
    path = find_visual_path(modified_parts, visual_id)
    visual_data = json.loads(modified_parts[path].decode("utf-8"))
    configure_slicer(visual_data, config["title"], config["position"])
    modified_parts[path] = json.dumps(
        visual_data,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_paths.append(path)

for visual_id, config in KPI_CARDS.items():
    path = find_visual_path(modified_parts, visual_id)
    visual_data = json.loads(modified_parts[path].decode("utf-8"))
    configure_card(visual_data, config["title"], config["position"])
    modified_parts[path] = json.dumps(
        visual_data,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_paths.append(path)

for visual_id, config in LOWER_VISUALS.items():
    path = find_visual_path(modified_parts, visual_id)
    visual_data = json.loads(modified_parts[path].decode("utf-8"))
    configure_lower_visual(visual_data, config["title"], config["position"])
    modified_parts[path] = json.dumps(
        visual_data,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_paths.append(path)

changed_paths = [
    path
    for path in sorted(original_parts)
    if modified_parts.get(path) != original_parts[path]
]

if set(changed_paths) != set(expected_paths):
    raise RuntimeError(
        "Validation failed. Unexpected report definitions changed.\n"
        f"Expected: {sorted(expected_paths)}\n"
        f"Detected: {changed_paths}"
    )

for visual_id in SLICERS:
    path = find_visual_path(modified_parts, visual_id)
    visual_data = json.loads(modified_parts[path].decode("utf-8"))
    mode = (
        visual_data.get("visual", {})
        .get("objects", {})
        .get("data", [])[0]
        .get("properties", {})
        .get("mode", {})
        .get("expr", {})
        .get("Literal", {})
        .get("Value")
    )
    if mode != "'Dropdown'":
        raise RuntimeError(f"Dropdown validation failed for {visual_id}.")

for visual_id in KPI_CARDS:
    path = find_visual_path(modified_parts, visual_id)
    visual_data = json.loads(modified_parts[path].decode("utf-8"))
    font_size = (
        visual_data.get("visual", {})
        .get("objects", {})
        .get("labels", [])[0]
        .get("properties", {})
        .get("fontSize", {})
        .get("expr", {})
        .get("Literal", {})
        .get("Value")
    )
    if font_size != "28D":
        raise RuntimeError(f"KPI font validation failed for {visual_id}.")

rollback_request_body = encode_definition(original_parts)
update_request_body = encode_definition(modified_parts)

print("\nFILTER AND KPI FIX VALIDATION PASSED")
print("Slicer dropdowns configured: 2")
print("Slicer heights increased: Yes")
print("KPI value font size: 28")
print("KPI titles updated: 5")
print("Table width: 430")
print("Risk chart width: 810")
print("Fields changed: No")
print("Filters changed: No")
print("Semantic model changed: No")
print("Rollback stored: rollback_request_body")

request_headers = get_headers()
update_response = requests.post(
    UPDATE_URL,
    headers=request_headers,
    json=update_request_body,
    timeout=180,
)
print(f"Initial update status: {update_response.status_code}")
wait_for_operation(update_response, request_headers)

print("\nFILTERS KPI AND LAYOUT FIX COMPLETED")
print("Final status: Succeeded")
print("Updated visuals: 9")
print("Semantic model changed: No")
print("Lakehouse required: No")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
