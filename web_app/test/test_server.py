import unittest

from web_app.server import app


class EditionRoutingTests(unittest.TestCase):
    def _rules(self):
        return {str(rule) for rule in app.url_map.iter_rules()}

    def test_english_routes_are_unprefixed(self):
        rules = self._rules()
        self.assertIn("/", rules)
        self.assertIn("/about", rules)
        self.assertIn("/map/<int:cluster_id>", rules)

    def test_spanish_routes_are_prefixed(self):
        rules = self._rules()
        self.assertIn("/es/", rules)
        self.assertIn("/es/about", rules)
        self.assertIn("/es/map/<int:cluster_id>", rules)

    def test_spanish_about_renders(self):
        response = app.test_client().get("/es/about")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Cómo funciona", body)

    def test_english_about_renders(self):
        response = app.test_client().get("/about")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("How", body)


if __name__ == "__main__":
    unittest.main()
