#!/usr/bin/env python3
"""
Unit tests for the Bruker spectroscopy reader and NIfTI-MRS writer.

Tests use synthetic Bruker scan directories rather than real data, so they run
without the CPZ study on disk. The layout they build mirrors ParaVision 360.3:
a ``rawdata.job0`` ordered (average, coil, point), the water reference embedded
in ``method`` as ``PVM_RefScan``, and ``GRPDLY`` in ``acqus``.
"""

import numpy as np
import pytest

from neurofaune.preprocess.utils.mrs.bruker_mrs import (
    apply_ppm_reference_shift,
    find_press_scans,
    measure_metabolite_offset,
    measure_water_ppm_offset,
    read_bruker_svs,
    read_voxel_geometry,
    remove_group_delay,
    resolve_group_delay,
    select_svs_scan,
    write_nifti_mrs,
)
from neurofaune.preprocess.utils.mrs.bruker_params import read_jcampdx, read_scan_params

N_POINTS = 256
N_COILS = 2
N_AVERAGES = 8
GROUP_DELAY = 12.25
DWELLTIME = 3e-4
SPEC_FREQ = 300.32


def make_fid(n_points, peaks, dwelltime=DWELLTIME, decay=40.0, group_delay=0.0):
    """Build a synthetic FID from (frequency_hz, amplitude) peaks."""
    time = np.arange(n_points) * dwelltime
    fid = np.zeros(n_points, dtype=complex)
    for frequency, amplitude in peaks:
        fid += amplitude * np.exp(2j * np.pi * frequency * time - time * decay)
    if group_delay:
        # Delay by shifting the other way round from remove_group_delay.
        ramp = np.exp(-2j * np.pi * np.fft.fftfreq(n_points) * group_delay)
        fid = np.fft.ifft(np.fft.fft(fid) * ramp)
    return fid


def write_jcampdx(path, entries):
    """Write a minimal JCAMP-DX parameter file."""
    lines = ['##TITLE=synthetic', '$$ generated for tests']
    for key, value in entries.items():
        if isinstance(value, str):
            lines.append(f'##${key}={value}')
        elif isinstance(value, (int, float)):
            lines.append(f'##${key}={value}')
        else:
            array = np.asarray(value)
            shape = ', '.join(str(n) for n in array.shape)
            lines.append(f'##${key}=( {shape} )')
            lines.append(' '.join(f'{float(v):.17g}' for v in array.ravel()))
    lines.append('##END=')
    path.write_text('\n'.join(lines) + '\n')


@pytest.fixture
def press_scan(tmp_path):
    """A synthetic water-suppressed PRESS scan directory."""
    scan_dir = tmp_path / 'session' / '13'
    scan_dir.mkdir(parents=True)

    # Metabolite data: one peak per average, identical across coils apart from
    # a fixed gain difference, so coil ordering is detectable.
    metab = np.empty((N_AVERAGES, N_COILS, N_POINTS), dtype=complex)
    for average in range(N_AVERAGES):
        base = make_fid(N_POINTS, [(-500.0, 1000.0)], group_delay=GROUP_DELAY)
        for coil in range(N_COILS):
            metab[average, coil] = base * (1.0 if coil == 0 else 0.5)

    interleaved = np.empty(metab.size * 2)
    flat = metab.ravel()
    interleaved[0::2] = flat.real
    interleaved[1::2] = flat.imag
    (scan_dir / 'rawdata.job0').write_bytes(
        np.round(interleaved * 1000).astype('<i4').tobytes()
    )

    # Water reference: on-resonance, both coils, same group delay.
    water = make_fid(N_POINTS, [(0.0, 50000.0)], decay=10.0, group_delay=GROUP_DELAY)
    ref = np.empty((N_COILS, N_POINTS * 2))
    for coil in range(N_COILS):
        ref[coil, 0::2] = (water * (1.0 if coil == 0 else 0.5)).real
        ref[coil, 1::2] = (water * (1.0 if coil == 0 else 0.5)).imag

    write_jcampdx(scan_dir / 'method', {
        'Method': '<Bruker:PRESS>',
        'PVM_EchoTime': 20,
        'PVM_RepetitionTime': 2000,
        'PVM_NAverages': N_AVERAGES,
        'PVM_EncNReceivers': N_COILS,
        'PVM_SpecMatrix': np.array([N_POINTS]),
        'PVM_SpecSWH': np.array([1.0 / DWELLTIME]),
        'PVM_WsOnOff': 'On',
        'PVM_WsMode': 'VAPOR',
        'PVM_Nucleus1': '<1H>',
        'PVM_FrqRefPpm': np.array([4.7, 0.0]),
        'PVM_FrqWork': np.array([SPEC_FREQ, 0.0]),
        'PVM_VoxArrSize': np.array([[7.5, 2.0, 2.0]]),
        'PVM_VoxArrPosition': np.array([[0.0, 5.0, 0.0]]),
        'PVM_VoxArrGradOrient': np.eye(3).reshape(1, 3, 3),
        'PVM_VoxelGeoCub': '(((1 0 0 0 1 0 0 0 1, 0 5 0, 0 0 0), 7.5 2 2), 1)',
        'PVM_RefScan': ref,
    })
    write_jcampdx(scan_dir / 'acqp', {'SFO1': SPEC_FREQ, 'ACQ_scan_name': '<PRESS_test>'})
    write_jcampdx(scan_dir / 'acqus', {'GRPDLY': GROUP_DELAY})
    return scan_dir


@pytest.fixture
def shim_scan(press_scan):
    """An unsuppressed single-average PRESS shim scan alongside the real one."""
    scan_dir = press_scan.parent / '6'
    scan_dir.mkdir()
    for name in ('method', 'acqp', 'acqus'):
        text = (press_scan / name).read_text()
        text = text.replace('##$PVM_WsOnOff=On', '##$PVM_WsOnOff=Off')
        text = text.replace(f'##$PVM_NAverages={N_AVERAGES}', '##$PVM_NAverages=1')
        (scan_dir / name).write_text(text)
    return scan_dir


class TestJcampdxParsing:
    def test_scalars_arrays_and_strings(self, press_scan):
        params = read_jcampdx(press_scan / 'method')
        assert params['Method'] == 'Bruker:PRESS'
        assert params['PVM_EchoTime'] == 20
        assert params['PVM_WsMode'] == 'VAPOR'
        assert params['PVM_VoxArrSize'].shape == (1, 3)

    def test_run_length_encoding_is_expanded(self, tmp_path):
        path = tmp_path / 'reco'
        path.write_text('##$RECO_transposition=( 41 )\n@41*(1)\n##END=\n')
        value = read_jcampdx(path)['RECO_transposition']
        assert value.shape == (41,)
        assert np.all(value == 1)

    def test_method_overrides_acqp(self, press_scan):
        # Both files are read; method holds the sequence-level view.
        params = read_scan_params(press_scan)
        assert params['Method'] == 'Bruker:PRESS'
        assert params['SFO1'] == SPEC_FREQ


class TestGroupDelay:
    def test_integer_part_comes_from_the_data(self):
        # GRPDLY understates the delay by a point on PV-360, so the resolver
        # must follow the echo top rather than the parameter. Delay by a whole
        # number of points so the top falls unambiguously on one sample.
        delay = 14.0
        fid = make_fid(N_POINTS, [(0.0, 1.0)], group_delay=delay)[:, None, None]
        assert np.abs(fid[:, 0, 0]).argmax() == delay

        resolved = resolve_group_delay(fid, GROUP_DELAY)
        # Integer part from the data (14), fractional part from GRPDLY (.25).
        assert resolved == pytest.approx(delay + GROUP_DELAY % 1, abs=1e-9)

    def test_fractional_part_comes_from_grpdly(self):
        fid = make_fid(N_POINTS, [(0.0, 1.0)], group_delay=10.0)[:, None, None]
        for grpdly in (9.1, 10.6, 11.75):
            resolved = resolve_group_delay(fid, grpdly)
            assert resolved % 1 == pytest.approx(grpdly % 1, abs=1e-9)

    def test_removal_puts_the_echo_top_first(self):
        fid = make_fid(N_POINTS, [(0.0, 1.0)], group_delay=GROUP_DELAY)[:, None, None]
        corrected = remove_group_delay(fid, GROUP_DELAY)
        profile = np.abs(corrected[:, 0, 0])
        assert profile.argmax() == 0

    def test_zero_delay_is_a_no_op(self):
        fid = make_fid(N_POINTS, [(0.0, 1.0)])[:, None, None]
        assert np.array_equal(remove_group_delay(fid, 0.0), fid)


class TestScanSelection:
    def test_finds_both_press_scans(self, press_scan, shim_scan):
        scans = find_press_scans(press_scan.parent)
        assert [s['scan_number'] for s in scans] == [6, 13]

    def test_selects_the_water_suppressed_acquisition(self, press_scan, shim_scan):
        assert select_svs_scan(press_scan.parent) == press_scan

    def test_returns_none_without_a_suppressed_scan(self, shim_scan, press_scan):
        import shutil

        shutil.rmtree(press_scan)
        assert select_svs_scan(shim_scan.parent) is None

    def test_skips_an_aborted_scan(self, press_scan):
        # A later scan aborted at the console keeps its parameter files but has
        # no data; an earlier complete scan must win rather than be lost.
        aborted = press_scan.parent / '19'
        aborted.mkdir()
        for name in ('method', 'acqp', 'acqus'):
            (aborted / name).write_text((press_scan / name).read_text())

        assert [s['scan_number'] for s in find_press_scans(press_scan.parent)] == [13, 19]
        assert select_svs_scan(press_scan.parent) == press_scan

    def test_returns_none_when_every_scan_is_aborted(self, press_scan):
        (press_scan / 'rawdata.job0').unlink()
        assert select_svs_scan(press_scan.parent) is None


class TestReader:
    def test_shape_and_ordering(self, press_scan):
        svs = read_bruker_svs(press_scan)
        assert svs.n_coils == N_COILS
        assert svs.n_averages == N_AVERAGES
        assert svs.source == 'rawdata'
        # Coil 1 was written at half the gain of coil 0; if the (average, coil)
        # axes were swapped the two would instead be indistinguishable.
        power = np.abs(svs.metab).mean(axis=(0, 2))
        assert power[0] / power[1] == pytest.approx(2.0, rel=0.05)

    def test_acquisition_parameters(self, press_scan):
        svs = read_bruker_svs(press_scan)
        assert svs.echo_time == pytest.approx(0.020)
        assert svs.repetition_time == pytest.approx(2.0)
        assert svs.dwelltime == pytest.approx(DWELLTIME)
        assert svs.spectrometer_frequency == pytest.approx(SPEC_FREQ)
        assert svs.nucleus == '1H'

    def test_water_reference_keeps_coils(self, press_scan):
        svs = read_bruker_svs(press_scan)
        assert svs.water_ref.shape[1] == N_COILS

    def test_metabolite_and_reference_stay_the_same_length(self, press_scan):
        # The group delay must be resolved once and applied to both. Resolving
        # each separately can round to different integers, leaving the arrays a
        # point apart -- which fsl_mrs_preproc only discovers at the
        # eddy-current step, as a broadcast error.
        svs = read_bruker_svs(press_scan)
        assert svs.metab.shape[0] == svs.water_ref.shape[0]

    def test_same_length_when_echo_tops_differ(self, press_scan):
        # Give the water reference a top one sample later than the metabolite
        # FID, which is what happens on real sessions.
        water = make_fid(N_POINTS, [(0.0, 50000.0)], decay=10.0,
                         group_delay=GROUP_DELAY + 1)
        ref = np.empty((N_COILS, N_POINTS * 2))
        for coil in range(N_COILS):
            ref[coil, 0::2] = water.real
            ref[coil, 1::2] = water.imag

        text = (press_scan / 'method').read_text().split('##$PVM_RefScan=')[0]
        shape = f'( {N_COILS}, {N_POINTS * 2} )'
        values = ' '.join(f'{float(v):.17g}' for v in ref.ravel())
        (press_scan / 'method').write_text(
            f'{text}##$PVM_RefScan={shape}\n{values}\n##END=\n')

        svs = read_bruker_svs(press_scan)
        assert svs.metab.shape[0] == svs.water_ref.shape[0]

    def test_rejects_a_non_press_scan(self, press_scan):
        text = (press_scan / 'method').read_text().replace(
            '##$Method=<Bruker:PRESS>', '##$Method=<Bruker:RARE>')
        (press_scan / 'method').write_text(text)
        with pytest.raises(ValueError, match='not a Bruker:PRESS'):
            read_bruker_svs(press_scan)

    def test_falls_back_when_rawdata_is_missing(self, press_scan):
        (press_scan / 'rawdata.job0').unlink()
        proc_dir = press_scan / 'pdata' / '1'
        proc_dir.mkdir(parents=True)
        fid = make_fid(N_POINTS, [(-500.0, 1.0)], group_delay=GROUP_DELAY)
        interleaved = np.empty(N_POINTS * 2)
        interleaved[0::2], interleaved[1::2] = fid.real, fid.imag
        (proc_dir / 'fid_proc.64').write_bytes(interleaved.astype('<f8').tobytes())

        svs = read_bruker_svs(press_scan)
        assert svs.source == 'bruker_averaged'
        assert svs.n_coils == 1

    def test_raises_when_no_data_is_readable(self, press_scan):
        (press_scan / 'rawdata.job0').unlink()
        with pytest.raises(ValueError, match='No usable FID data'):
            read_bruker_svs(press_scan)


class TestVoxelGeometry:
    """PVM_VoxArrPosition is in the voxel's rotated frame, not magnet coords."""

    def _geocub(self, angle_deg, position):
        a = np.deg2rad(angle_deg)
        rotation = [np.cos(a), np.sin(a), 0, -np.sin(a), np.cos(a), 0, 0, 0, 1]
        nums = ' '.join(f'{v:.10f}' for v in rotation)
        pos = ' '.join(f'{v:.10f}' for v in position)
        return {'PVM_VoxelGeoCub': f'((({nums}, {pos}, 0 0 0), 7.5 2 2), 1)'}

    def test_reads_rotation_and_position(self):
        rotation, position = read_voxel_geometry(self._geocub(10.0, (-1.1, 8.0, 1.4)))
        assert position == pytest.approx([-1.1, 8.0, 1.4])
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        assert np.degrees(np.arctan2(rotation[0, 1], rotation[0, 0])) == pytest.approx(10.0)

    def test_position_differs_from_the_rotated_field(self):
        # The regression this guards: R @ geocub_position is what Bruker
        # reports as PVM_VoxArrPosition, so using that field directly
        # misplaces the voxel in proportion to the rotation -- 1.7 mm at the
        # 12 degrees of the most angled cuprizone session, and nothing at 0.
        for angle, expected_error in ((0.0, 0.0), (10.0, 1.4)):
            rotation, position = read_voxel_geometry(self._geocub(angle, (-1.1, 8.0, 1.4)))
            rotated = rotation @ position
            assert np.linalg.norm(rotated - position) == pytest.approx(
                expected_error, abs=0.15)

    def test_missing_geometry_object_is_an_error(self):
        # There is no safe fallback: the alternative field is in the wrong frame.
        with pytest.raises(KeyError, match='PVM_VoxelGeoCub'):
            read_voxel_geometry({'PVM_VoxArrPosition': np.array([[0.0, 5.0, 0.0]])})

    def test_unparseable_geometry_object_is_an_error(self):
        with pytest.raises(ValueError, match='PVM_VoxelGeoCub'):
            read_voxel_geometry({'PVM_VoxelGeoCub': '(((1 0 0)))'})


class TestPpmReferencing:
    def test_measures_a_known_water_offset(self):
        offset_hz = 15.0
        water = make_fid(N_POINTS, [(offset_hz, 1.0)], decay=5.0)[:, None, None]
        measured = measure_water_ppm_offset(water, SPEC_FREQ, DWELLTIME)
        # ppm = fftshift(fftfreq) / f0 + reference, so ppm runs with frequency.
        assert measured == pytest.approx(offset_hz / SPEC_FREQ, abs=0.005)

    def test_correction_moves_water_to_its_true_shift(self):
        offset_hz = 15.0
        water = make_fid(N_POINTS, [(offset_hz, 1.0)], decay=5.0)[:, None, None]
        metab = make_fid(N_POINTS, [(-500.0, 1.0)])[:, None, None]

        _, shifted, applied = apply_ppm_reference_shift(
            metab, water, water_ppm=4.7, reference_ppm=4.65,
            spectrometer_frequency=SPEC_FREQ, dwelltime=DWELLTIME,
        )
        # Water ends up at 4.7 on an axis whose carrier reads 4.65, i.e. 0.05
        # above the carrier.
        residual = measure_water_ppm_offset(shifted, SPEC_FREQ, DWELLTIME)
        assert residual == pytest.approx(4.7 - 4.65, abs=0.01)
        assert applied == pytest.approx(0.05 - offset_hz / SPEC_FREQ, abs=0.01)

    def test_no_reference_means_no_shift(self):
        metab = make_fid(N_POINTS, [(-500.0, 1.0)])[:, None, None]
        out, ref, applied = apply_ppm_reference_shift(
            metab, None, 4.7, 4.65, SPEC_FREQ, DWELLTIME)
        assert applied == 0.0
        assert ref is None
        assert np.array_equal(out, metab)

    def _two_singlet_fid(self, tcr_ppm, naa_ppm, spec_freq=SPEC_FREQ):
        """A spectrum with tCr and NAA singlets at the given chemical shifts."""
        from neurofaune.preprocess.utils.mrs.bruker_mrs import FSL_MRS_REFERENCE_PPM

        peaks = [((ppm - FSL_MRS_REFERENCE_PPM) * spec_freq, amp)
                 for ppm, amp in ((tcr_ppm, 1.0), (naa_ppm, 1.2))]
        return make_fid(1024, peaks, decay=8.0)[:, None, None]

    def test_metabolite_offset_measures_the_displacement(self):
        # tCr sits 0.09 ppm low, as the CPZ data does before referencing.
        fid = self._two_singlet_fid(3.027 - 0.09, 2.008 - 0.09)
        offset = measure_metabolite_offset(fid, None, SPEC_FREQ, DWELLTIME)
        assert offset == pytest.approx(0.09, abs=0.01)

    def test_metabolite_offset_is_zero_when_already_referenced(self):
        fid = self._two_singlet_fid(3.027, 2.008)
        offset = measure_metabolite_offset(fid, None, SPEC_FREQ, DWELLTIME)
        assert offset == pytest.approx(0.0, abs=0.01)

    def test_metabolite_offset_rejects_a_wrong_separation(self):
        # Peaks the expected distance apart are what makes a wide search window
        # safe; without that check a misidentified peak would be baked in.
        fid = self._two_singlet_fid(3.30, 1.75)
        assert measure_metabolite_offset(fid, None, SPEC_FREQ, DWELLTIME) is None

    def test_reader_can_skip_referencing(self, press_scan):
        svs = read_bruker_svs(press_scan, reference_ppm=None)
        assert svs.ppm_correction == 0.0


class TestNiftiMrsOutput:
    def test_written_file_round_trips(self, press_scan, tmp_path):
        pytest.importorskip('nifti_mrs')
        import nibabel as nib

        svs = read_bruker_svs(press_scan)
        outputs = write_nifti_mrs(svs, tmp_path, 'sub-test_ses-1')

        image = nib.load(outputs['svs'])
        assert image.shape == (1, 1, 1, svs.n_points, N_COILS, N_AVERAGES)
        assert image.header['pixdim'][4] == pytest.approx(DWELLTIME)
        # The NIfTI-MRS header extension must be present or FSL-MRS cannot read it.
        assert [e.get_code() for e in image.header.extensions] == [44]

    def test_declares_a_version_fsl_can_read(self, press_scan, tmp_path):
        pytest.importorskip('nifti_mrs')
        import nibabel as nib

        svs = read_bruker_svs(press_scan)
        outputs = write_nifti_mrs(svs, tmp_path, 'sub-test_ses-1')
        intent = nib.load(outputs['svs']).header['intent_name'].item().decode()
        # FSL 6.0.7 gates on float(version) < 0.2, so v0.10+ is rejected.
        assert float(intent.replace('mrs_v', '').replace('_', '.')) >= 0.2

    def test_affine_encodes_the_scaled_voxel(self, press_scan, tmp_path):
        pytest.importorskip('nifti_mrs')
        import nibabel as nib

        svs = read_bruker_svs(press_scan)
        outputs = write_nifti_mrs(svs, tmp_path, 'sub-test_ses-1')
        affine = nib.load(outputs['svs']).affine
        # Voxel orientation is identity here, so the diagonal is the 10x size.
        assert np.diag(affine)[:3] == pytest.approx([75.0, 20.0, 20.0])
        assert affine[:3, 3] == pytest.approx([0.0, 50.0, 0.0])
