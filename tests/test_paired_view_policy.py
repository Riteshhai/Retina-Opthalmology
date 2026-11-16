"""Synthetic checks for the initial-view acquisition boundary and slot accounting."""

import numpy as np
import pytest

from qsart.paired_view_policy import ValueModel, budget_routes, first_view_features, fit_value


def test_rate_and_positive_cost_are_applied_to_initial_view_features():
    features = first_view_features([[.7, .1, .1, .05, .05]] * 10, np.linspace(.5, 1., 10))
    fixed = ValueModel((0.,) * 5, (1.,) * 5, (.4, 0., 0., 0., 0.), 0.)
    rows = {row['id']: row for row in budget_routes(features, fixed)}
    assert len(rows) == 26
    assert not any(rows['first_view:0']['selected']) and all(rows['both_views:5']['selected'])
    assert rows['voi:3']['slots'] == 5
    assert rows['voi:3']['selected'] == [True] * 5 + [False] * 5
    # The highest gross gain is exactly the acquisition price, so it is not positive net value.
    assert rows['voi_positive:3']['selected'] == [False] * 10
    assert rows['entropy:3']['selected'] == [True] * 5 + [False] * 5
    assert rows['random:3'] == {row['id']: row for row in budget_routes(features, fixed)}['random:3']


def test_fit_uses_frozen_initial_features_and_centered_targets():
    features = first_view_features([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]], [.25, .75])
    np.testing.assert_allclose(features, [[.75, 0., 0., 0., 0.], [.25, 0., 1., 0., 1.]])
    target = np.array([-1., 3.])
    fit = fit_value(features, target)
    assert fit.intercept == 1.
    assert fit.feature_scale[1] == 1.
    np.testing.assert_allclose(fit.predict(features).mean(), target.mean())
    with pytest.raises(ValueError, match='probabilities'):
        first_view_features([[.2] * 5], [float('nan')])
