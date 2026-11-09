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
