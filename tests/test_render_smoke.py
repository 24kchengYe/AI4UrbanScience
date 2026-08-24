from ai4us.figures.render import render_selected


def test_documented_module_renderer(tmp_path) -> None:
    output = tmp_path / "render"
    report = render_selected(["4", "S03", "S11"], output)
    assert report["figure_count"] == 3
    assert report["records"][0]["status"] == "PASS"
    files = list((output / "figure4").glob("*"))
    assert {path.suffix for path in files} == {".pdf", ".svg", ".png"}
    for supplementary in report["records"][1:]:
        assert supplementary["diagnostic_selection"] == {
            "mode": "all_released_rows",
            "selected_rows": supplementary["source_data_rows"],
            "omitted_rows": 0,
        }
