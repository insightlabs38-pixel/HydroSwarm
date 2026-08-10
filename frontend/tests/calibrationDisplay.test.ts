import {
  calibrationStatusText,
  candidateCoverageLabel,
  candidateCoverageValueText,
  isCalibrationApplicable,
  measuredCoverageValueText,
} from '../src/calibrationDisplay';
import { demoIncident } from '../src/demoFixture';
import type { IncidentView } from '../src/types';

function withOverrides(overrides: Partial<IncidentView>): IncidentView {
  return { ...demoIncident, ...overrides };
}

test('isCalibrationApplicable defaults to true when the field is absent (LIVE/DEMO_FALLBACK unaffected)', () => {
  const incident = withOverrides({ calibrationApplicable: undefined });
  expect(isCalibrationApplicable(incident)).toBe(true);
});

test('isCalibrationApplicable is false only when explicitly set to false', () => {
  const incident = withOverrides({ calibrationApplicable: false });
  expect(isCalibrationApplicable(incident)).toBe(false);
});

test('calibrationStatusText reports valid/invalid normally, and "not applicable" for REFERENCE', () => {
  expect(calibrationStatusText(withOverrides({ calibrationValid: true }))).toBe('valid');
  expect(calibrationStatusText(withOverrides({ calibrationValid: false }))).toBe('invalid');
  expect(
    calibrationStatusText(withOverrides({ calibrationValid: true, calibrationApplicable: false })),
  ).toBe('Not applicable to reference replay');
});

test('candidateCoverageLabel switches from "Conformal target" to "Region target" when not applicable', () => {
  expect(candidateCoverageLabel(withOverrides({}))).toBe('Conformal target');
  expect(candidateCoverageLabel(withOverrides({ calibrationApplicable: false }))).toBe(
    'Region target',
  );
});

test('candidateCoverageValueText never presents a reference criterion as a real conformal percentage', () => {
  expect(candidateCoverageValueText(withOverrides({ candidateCoverage: 0.9 }))).toBe('90%');
  expect(
    candidateCoverageValueText(
      withOverrides({ candidateCoverage: 0.9, calibrationApplicable: false }),
    ),
  ).toBe('90% reference criterion');
});

test('measuredCoverageValueText reports "Not applicable" rather than "not measured" for REFERENCE', () => {
  expect(measuredCoverageValueText(withOverrides({ measuredCoverage: undefined }))).toBe(
    'not measured',
  );
  expect(measuredCoverageValueText(withOverrides({ measuredCoverage: 0.91 }))).toBe('91%');
  expect(
    measuredCoverageValueText(
      withOverrides({ measuredCoverage: 0.91, calibrationApplicable: false }),
    ),
  ).toBe('Not applicable');
});
