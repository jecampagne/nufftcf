# Usage

## Function signatures

All ACF functions share the same signature:

```python
c, b = fn(lags, t, x)                     # no-kernel ("regular") variant
c, b = fn(lags, t, x, bin_width=0.5)      # gaussian / rectangle variants
```

- `lags` — array of lag values (same units as `t`)
- `t` — sample times, sorted ascending (float array, e.g. days since start)
- `x` — signal values, same length as `t`
- `bin_width` — kernel half-width (gaussian σ or rectangle half-width), same units as `t`
- Returns `(c, b)` — ACF estimate and effective pair count, both shape `(len(lags),)`

The CCF functions take a second `(s, y)` series in addition to `(t, x)`:

```python
c, b = fn(lags, t, x, s, y, bin_width=0.5)
```

- `s` — sample times of the second series, sorted ascending, on the **same time
  origin as `t`** (see the note on the common time origin on the [Home](index.md) page)
- `y` — signal values of the second series, same length as `s`
- Returns `(c, b)` — CCF estimate (Pearson-normalised, `c ~ 1` at perfect correlation)
  and effective pair count, both shape `(len(lags),)`. By convention a positive lag
  means `y` lags behind `x` (the CCF peaks at `lag = tau0` when `y(t) ~ x(t - tau0)`)

--8<-- "README.md:guide"
