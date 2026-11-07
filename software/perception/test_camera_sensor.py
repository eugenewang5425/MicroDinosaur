"""Reboot-mode regression checks using media-control readbacks, no hardware."""
from types import SimpleNamespace
import unittest

from camera_sensor import configure_sensor, read_format


class Media:
    def __init__(self, size='3280x2464', reject=False):
        self.size = size
        self.reject = reject
        self.calls = []
        self.formats = {}

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        if '--get-v4l2' in args:
            pad = args[-1]
            code = 'YUYV8_2X8' if pad == '"rkisp-isp-subdev":2' else 'SRGGB10_1X10'
            value = self.formats.get(pad, self.size)
            return SimpleNamespace(returncode=0, stdout=f'[fmt:{code}/{value} field:none]', stderr='')
        if '-V' in args and not self.reject:
            self.formats[args[-1].split('[')[0]] = '1920x1080'
        return SimpleNamespace(returncode=0, stdout='', stderr='')


class SensorTests(unittest.TestCase):
    def test_reboot_default_is_restored_and_all_links_verified(self):
        media = Media()
        result = configure_sensor(run=media)
        self.assertTrue(result['changed'])
        self.assertEqual(result['actual'], ['SRGGB10_1X10', 1920, 1080])
        self.assertEqual(sum('-V' in c for c in media.calls), 5)
        self.assertIn(['systemctl', 'restart', 'rkaiq_3A'], media.calls)

    def test_already_configured_has_no_mutation(self):
        media = Media('1920x1080')
        self.assertFalse(configure_sensor(run=media)['changed'])
        self.assertFalse(any('-V' in c or 'systemctl' in c for c in media.calls))

    def test_partial_link_configuration_is_repaired(self):
        media = Media('1920x1080')
        media.formats['"rkisp-isp-subdev":2'] = '3280x2464'
        self.assertTrue(configure_sensor(run=media)['changed'])

    def test_explicit_current_mode_does_not_write(self):
        media = Media()
        result = configure_sensor('current', run=media)
        self.assertEqual(result['actual'][1:], [3280, 2464])
        self.assertFalse(any('-V' in c or 'systemctl' in c for c in media.calls))

    def test_silent_driver_rejection_fails_readback(self):
        with self.assertRaisesRegex(RuntimeError, 'readback mismatch'):
            configure_sensor(run=Media(reject=True))

    def test_command_error_is_not_accepted(self):
        def fail(*args, **kwargs):
            return SimpleNamespace(returncode=1, stdout='', stderr='device busy')
        with self.assertRaisesRegex(RuntimeError, 'device busy'):
            configure_sensor(run=fail)

    def test_malformed_format_is_not_accepted(self):
        with self.assertRaises(RuntimeError):
            read_format('unrecognized')


if __name__ == '__main__':
    unittest.main()
