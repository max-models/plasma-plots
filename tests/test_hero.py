import numpy as np
import pytest

from plasma_plots.hero import HeroFigure


def test_marks_and_tracers_render():
    fig = HeroFigure(xlim=(0, 2), ylim=(-1, 1), width=200, height=100)
    x = np.linspace(0, 2, 5)
    line = fig.line(x, np.sin(x), opacity=0.5, dash="2 3", title="a <line>")
    fig.tracer(line, duration=3, offset=1)
    fig.tracer([0, 1, 2], y=[0, 1, 0], duration=2, tone="accent")
    fig.circle(1, 0, 3).animate("opacity", [0, 1, 0], duration=2, begin=0.5)
    fig.circle(1, 0, 3, fill=False, opacity=0.4)
    fig.disc(0.5, 0, 0.1)
    fig.rect(0, 0, 1, 1, tone="low", opacity=0.3, outline="muted")
    svg = fig.to_svg()
    assert svg.startswith("<svg") and svg.count("<animateMotion") == 2
    assert 'd="M0 100 L50 ' not in svg  # y points up
    assert "M0 50" in svg and "a &lt;line&gt;" in svg
    assert 'stroke-dasharray="2 3"' in svg and 'class="f-low s-muted"' in svg
    assert 'r="10"' in svg  # a disc's radius is in data units


def test_save(tmp_path):
    fig = HeroFigure()
    fig.line([0, 1], [0, np.nan, 1][:2])
    path = fig.save(tmp_path / "assets" / "hero.svg")
    assert path.read_text().startswith("<svg")


def test_errors():
    fig = HeroFigure()
    with pytest.raises(ValueError, match="tone"):
        fig.line([0, 1], [0, 1], tone="red")
    with pytest.raises(ValueError, match="finite"):
        fig.line([np.nan], [np.nan])
    with pytest.raises(ValueError, match="trajectory"):
        fig.tracer(fig.circle(0, 0, 1), duration=1)
