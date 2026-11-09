"""extractor: a patient listed twice by CCDI ('pt-abc' and 'PT_ABC') is fetched once; repeated
clinical rows no longer stop the pivot ("Index contains duplicate entries, cannot reshape")."""
import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def extractor(monkeypatch):
    pytest.importorskip("bravado")
    pytest.importorskip("zarr")
    from bravado.client import SwaggerClient
    # module import builds portal clients from the live swagger specs: no network in tests
    class _Offline:                      # a SwaggerClient lists only its resources in dir()
        def __dir__(self):
            return []
    monkeypatch.setattr(SwaggerClient, "from_url", classmethod(lambda cls, *a, **k: _Offline()))
    sys.modules.pop("src.extractor", None)
    return importlib.import_module("src.extractor")


class _Call:
    def __init__(self, out):
        self.out = out

    def result(self):
        return self.out


def _client(calls, rows_for):
    def patient(studyId, patientId, **kw):
        calls.append(patientId)
        return _Call([SimpleNamespace(patientId=patientId, clinicalAttributeId=a, value=v) for a, v in rows_for(patientId)])

    def sample(studyId, sampleId, **kw):
        calls.append(sampleId)
        return _Call([SimpleNamespace(sampleId=sampleId, patientId="PT_ABC", clinicalAttributeId="CANCER_TYPE",
                                      value="Low-grade glioma")] * 2)            # portal repeats a row
    return SimpleNamespace(Clinical_Data=SimpleNamespace(getAllClinicalDataOfPatientInStudyUsingGET=patient,
                                                         getAllClinicalDataOfSampleInStudyUsingGET=sample))


def test_patient_listed_twice_is_fetched_once(extractor):
    calls = []
    cli = _client(calls, lambda p: [("SEX", "Male"), ("OS_MONTHS", "12")])
    df, ok = extractor.fetch_patient_clinical(cli, "pbta_all", ["pt-abc", "PT_ABC", "TARGET-50-X"])
    assert calls == ["PT_ABC", "TARGET-50-X"] and ok == {"PT_ABC", "TARGET-50-X"}
    assert sorted(df.patientId) == ["PT_ABC", "TARGET-50-X"] and set(df.columns) == {"patientId", "OS_MONTHS", "SEX"}


def test_repeated_attribute_rows_keep_the_first(extractor, capsys):
    rows = lambda p: [("SEX", "Male"), ("SEX", "Male"), ("RACE", "White"), ("RACE", "Asian")]
    df, _ = extractor.fetch_patient_clinical(_client([], rows), "s", ["PT_ABC"])
    assert df.loc[0, "SEX"] == "Male" and df.loc[0, "RACE"] == "White"
    assert "2 repeated attribute rows dropped (1 attributes with differing values: first kept)" in capsys.readouterr().out
    calls = []
    sam = extractor.fetch_sample_clinical(_client(calls, rows), "s", ["7316-1", "7316-1", "bs-xyz", "BS_XYZ"])
    assert calls == ["7316-1", "BS_XYZ"] and sam.set_index("sampleId").loc["7316-1", "patientId"] == "PT_ABC"


def test_samples_fall_back_to_the_portal_when_ccdi_lists_none_of_them(extractor, monkeypatch):
    monkeypatch.setattr(extractor, "_SAMPLE_SCOPE", "ccdi")     # default, whatever config/pipeline_config.yaml sets
    portal = ["7316-1", "7316-2", "7316-3"]
    # PBTA: CCDI's portal-form IDs belong to other studies -> every portal sample of the matched patients
    ids, note = extractor.select_samples("Glioma", ["SJHGG01", "TH_9"], portal)
    assert ids == portal and "none of the 2 CCDI sample IDs" in note
    # TARGET Wilms tumor: CCDI lists some of the portal samples -> only those (the 114-patient cohort)
    assert extractor.select_samples("Wilms Tumor", ["TARGET-50-A-01", "X"], ["TARGET-50-A-01", "TARGET-50-B-01"]) == \
        (["TARGET-50-A-01"], "")
    assert extractor.select_samples("Glioma", None, portal) == (portal, "")
    monkeypatch.setattr(extractor, "_SAMPLE_SCOPE", {"Glioma": "patients"})
    assert extractor.select_samples("Glioma", ["7316-1"], portal)[0] == portal


def test_omics_fetched_in_chunks_give_the_same_matrix(extractor, monkeypatch):
    import numpy as np
    import pandas as pd
    samples = [f"S{i:02d}" for i in range(7)]
    genes = [("G1", 1), ("G2", 2), ("G3", 3)]
    calls = []

    def fake_fetch(api_base, headers, profile_id, sample_ids):
        calls.append(list(sample_ids))
        rows = [{"molecularProfileId": profile_id, "hugoGeneSymbol": g, "entrezGeneId": e, "sampleId": s,
                 "patientId": "P", "studyId": "x", "value": str(10 * e + int(s[1:]))}
                for s in sample_ids for g, e in genes if not (g == "G3" and s == "S05")]
        return pd.DataFrame(rows)

    monkeypatch.setattr(extractor, "fetch_profile_data", fake_fetch)
    one = extractor.fetch_profile_wide("u", {}, "p", samples, chunk=100)
    many = extractor.fetch_profile_wide("u", {}, "p", samples, chunk=3)
    assert [len(c) for c in calls] == [7, 3, 3, 1]
    assert list(many.columns) == ["hugoGeneSymbol", "entrezGeneId", *samples]
    pd.testing.assert_frame_equal(one, many)
    assert np.isnan(many.set_index("hugoGeneSymbol").loc["G3", "S05"])


def test_profile_fetch_retries_a_timeout(extractor, monkeypatch):
    import requests
    tries = []

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return [{"molecularProfileId": "p", "gene": {"hugoGeneSymbol": "G1"}, "entrezGeneId": 1,
                     "sampleId": "S1", "patientId": "P", "studyId": "x", "value": "1.5"}]

    def post(*a, **k):
        tries.append(1)
        if len(tries) == 1:
            raise requests.Timeout("read timed out")
        return _Resp()

    monkeypatch.setattr(extractor.requests, "post", post)
    monkeypatch.setattr(extractor.time, "sleep", lambda s: None)
    df = extractor.fetch_profile_data("u", {}, "p", ["S1"])
    assert len(tries) == 2 and df.loc[0, "value"] == "1.5"
