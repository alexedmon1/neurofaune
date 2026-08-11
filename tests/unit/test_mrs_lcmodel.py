#!/usr/bin/env python3
"""
Unit tests for the LCModel fitter interface.

These cover the file formats and parsing only -- they do not invoke the
``lcmodel`` binary, so they run anywhere.
"""

import numpy as np
import pytest

from neurofaune.preprocess.utils.mrs.lcmodel import (
    find_lcmodel,
    parse_table,
    read_license,
    write_control,
    write_raw,
)


class TestLicence:
    def test_reads_key_and_owner(self, tmp_path):
        path = tmp_path / 'license'
        path.write_text("key = 210387309\nowner = 'Example Lab'\n")
        assert read_license(path) == (210387309, 'Example Lab')

    def test_missing_licence_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='licence'):
            read_license(tmp_path / 'nope')


class TestRawFormat:
    def test_header_and_values(self, tmp_path):
        fid = np.array([1 + 2j, 3 - 4j])
        path = write_raw(fid, tmp_path / 'm.RAW', 'sub-1Y_ses-1', conjugate=False)
        lines = path.read_text().splitlines()

        assert lines[0].strip() == '$NMID'
        assert "ID='sub-1Y_ses-1'" in lines[1]
        assert '$END' in lines[2]
        # Two complex points, one per line, real then imaginary.
        assert len(lines) == 5
        assert [float(v) for v in lines[3].split()] == [1.0, 2.0]
        assert [float(v) for v in lines[4].split()] == [3.0, -4.0]

    def test_conjugates_by_default(self, tmp_path):
        # LCModel takes the opposite convention to NIfTI-MRS; getting this
        # wrong still produces a table, just a meaningless one.
        path = write_raw(np.array([1 + 2j]), tmp_path / 'm.RAW', 'x')
        values = [float(v) for v in path.read_text().splitlines()[3].split()]
        assert values == [1.0, -2.0]

    def test_scaling_is_applied(self, tmp_path):
        path = write_raw(np.array([1 + 0j]), tmp_path / 'm.RAW', 'x',
                         scale=1000.0, conjugate=False)
        values = [float(v) for v in path.read_text().splitlines()[3].split()]
        assert values[0] == pytest.approx(1000.0)


class TestControlFile:
    def _write(self, tmp_path, **kwargs):
        defaults = dict(
            output_file=tmp_path / 'c.control',
            raw_file=tmp_path / 'm.RAW',
            basis_file=tmp_path / 'b.basis',
            output_prefix=tmp_path / 'out',
            n_points=2048,
            dwelltime=3e-4,
            spectrometer_frequency=300.32,
            key=1234,
            owner='Lab',
        )
        defaults.update(kwargs)
        return write_control(**defaults).read_text()

    def test_required_fields(self, tmp_path):
        text = self._write(tmp_path)
        assert '$LCMODL' in text and '$END' in text
        assert 'KEY(1)=1234' in text
        assert "OWNER='Lab'" in text
        assert 'NUNFIL=2048' in text
        assert 'HZPPPM=300.320000' in text

    def test_ppm_range_is_ordered(self, tmp_path):
        # PPMST is the downfield (larger) end regardless of argument order.
        text = self._write(tmp_path, ppm_range=(4.2, 0.2))
        assert 'PPMST=4.200' in text
        assert 'PPMEND=0.200' in text

    def test_water_scaling_only_when_a_reference_is_given(self, tmp_path):
        assert 'DOWS' not in self._write(tmp_path)
        text = self._write(tmp_path, h2o_file=tmp_path / 'w.H2O')
        assert 'DOWS=T' in text
        # Eddy-current correction has already been applied upstream.
        assert 'DOECC=F' in text

    def test_extra_settings_win(self, tmp_path):
        text = self._write(tmp_path, extra={'NSIMUL': 0, 'PPMST': '4.000'})
        assert 'NSIMUL=0' in text
        assert 'PPMST=4.000' in text


class TestTableParsing:
    TABLE = """
 LCModel (Version 6.3-1L) Copyright
 Data shifted by  0.001 ppm

 Conc.  %SD   /Cr+PCr   Metabolite

  1.234E+00   3%    1.252  NAA
  9.870E-01   2%    1.000  Cr+PCr
  5.000E-01 999%    0.507  Glc
  2.000E-01   4%    0.203  GPC+PCh

 Some following section that should not be parsed
"""

    def test_parses_the_concentration_block(self, tmp_path):
        path = tmp_path / 'lcmodel.table'
        path.write_text(self.TABLE)
        table = parse_table(path).set_index('metabolite')

        assert list(table.index) == ['NAA', 'Cr+PCr', 'Glc', 'GPC+PCh']
        assert table.loc['NAA', 'concentration'] == pytest.approx(1.234)
        assert table.loc['NAA', 'crlb_percent'] == pytest.approx(3.0)
        assert table.loc['NAA', 'ratio_to_cr'] == pytest.approx(1.252)

    def test_keeps_unreliable_metabolites(self, tmp_path):
        # A 999% CRLB means "not measurable", but it should still be reported
        # so downstream filtering is explicit rather than silent.
        path = tmp_path / 'lcmodel.table'
        path.write_text(self.TABLE)
        table = parse_table(path).set_index('metabolite')
        assert table.loc['Glc', 'crlb_percent'] == pytest.approx(999.0)

    def test_empty_table(self, tmp_path):
        path = tmp_path / 'lcmodel.table'
        path.write_text('no concentrations here\n')
        assert parse_table(path).empty


class TestBinaryDiscovery:
    def test_explicit_directory_wins(self, tmp_path):
        binary = tmp_path / 'lcmodel'
        binary.write_text('#!/bin/sh\n')
        binary.chmod(0o755)
        assert find_lcmodel(tmp_path) == str(binary)

    def test_missing_binary_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            'neurofaune.preprocess.utils.mrs.lcmodel.DEFAULT_LCMODEL_DIRS',
            (tmp_path / 'absent',))
        monkeypatch.setattr(
            'neurofaune.preprocess.utils.mrs.lcmodel.shutil.which', lambda _: None)
        with pytest.raises(FileNotFoundError, match='lcmodel'):
            find_lcmodel()
