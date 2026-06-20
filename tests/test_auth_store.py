import json
import tempfile
import unittest
from pathlib import Path

from auth_store import signin_user, signup_user


class AuthStoreTests(unittest.TestCase):
    def test_signup_hashes_password_and_signin_verifies_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "users.json"

            success, message = signup_user(
                "test_user",
                "test@example.com",
                "strong-password",
                path=path,
            )

            self.assertTrue(success, message)
            record = json.loads(path.read_text(encoding="utf-8"))["test_user"]
            self.assertNotIn("password", record)
            self.assertIn("password_hash", record)
            self.assertTrue(signin_user("test_user", "strong-password", path=path))
            self.assertFalse(signin_user("test_user", "wrong-password", path=path))

    def test_legacy_plaintext_record_is_migrated_on_login(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "users.json"
            path.write_text(
                json.dumps(
                    {
                        "legacy": {
                            "email": "legacy@example.com",
                            "password": "old-password",
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(signin_user("legacy", "old-password", path=path))
            record = json.loads(path.read_text(encoding="utf-8"))["legacy"]
            self.assertNotIn("password", record)
            self.assertIn("password_hash", record)

    def test_signup_validation_rejects_weak_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "users.json"
            success, _ = signup_user("x", "invalid", "short", path=path)
            self.assertFalse(success)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
