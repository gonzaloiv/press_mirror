import unittest

from pipeline.clustering.labels import generate_topic_label


class GenerateTopicLabelTests(unittest.TestCase):
    def test_builds_labels_from_terms_shared_by_all_headlines(self):
        examples = [
            (
                [
                    "Moscow drone attack disrupts power grid",
                    "Drone attack targets Moscow",
                    "Russia faces drone strike on Moscow",
                ],
                "Moscow drone attack",
            ),
            (
                [
                    "Russia denies plague outbreak after lab worker death",
                    "Russia rejects plague outbreak following lab worker death",
                    "Russian officials deny outbreak linked to plague",
                ],
                "Russia denies plague outbreak",
            ),
            (
                [
                    "US bomber withdrawal from UK base",
                    "US bombers leave UK base",
                    "Bombers depart UK base as US aircraft returns",
                ],
                "US bombers leave UK base",
            ),
            (
                [
                    "Brazilian election: family rule vs. democracy",
                    "Brazilian election draws voters",
                ],
                "Brazilian election",
            ),
            (
                [
                    "French school strikes: why are upperclassmen so angry?",
                    "French school strikes continue",
                ],
                "French school strikes",
            ),
            (
                [
                    "Spain delays elections: Sanchez hopes voters punish housing crisis",
                    "Spain delays elections amid housing crisis",
                ],
                "Spain delays elections",
            ),
        ]

        for headlines, expected in examples:
            with self.subTest(headlines=headlines):
                label = generate_topic_label(headlines)
                self.assertEqual(label, expected)
                self.assertLessEqual(len(label.split()), 7)
                self.assertNotIn("?", label)
                self.assertNotIn("vs.", label.casefold())

    def test_fallback_removes_editorial_tail_and_obeys_word_limit(self):
        label = generate_topic_label(
            ["Brazilian Election: Family Rule vs. Democracy"]
        )

        self.assertEqual(label, "Brazilian Election")
        self.assertLessEqual(len(label.split()), 7)
        self.assertNotIn("?", label)
        self.assertNotIn("vs.", label.casefold())

    def test_non_english_cluster_gets_an_english_label(self):
        label = generate_topic_label(
            [
                "Huelga escolar en Francia",
                "Francia mantiene huelga escolar",
            ]
        )

        self.assertEqual(label, "French school protests")
        self.assertTrue(label.isascii())
        self.assertLessEqual(len(label.split()), 7)

    def test_mixed_language_cluster_uses_only_english_headlines(self):
        label = generate_topic_label(
            [
                "Guerra de drones en Moscú",
                "Russia denies drone attack on Moscow",
                "Russian drone attack targets Moscow",
            ]
        )

        self.assertEqual(label, "Moscow drone attack")

    def test_labels_keep_the_main_action_and_its_topic(self):
        examples = [
            (
                [
                    "Russia launches invasion into Ukraine",
                    "Russian invasion of Ukraine intensifies",
                    "Ukraine under Russian invasion",
                ],
                "Russian invasion of Ukraine",
            ),
            (
                [
                    "US bombers leave UK base as Iranian drones threaten region",
                    "US bomber withdrawal from UK base over Iranian drone threat",
                    "Bombers depart UK base after Iran drone threats",
                ],
                "US bombers leave UK base",
            ),
            (
                [
                    "Earthquake kills dozens in Turkey",
                    "Deadly earthquake strikes Turkey",
                    "Turkey earthquake death toll rises",
                ],
                "Turkey earthquake",
            ),
            (
                [
                    "Parliament votes to ban social media for children",
                    "Country approves social media ban for minors",
                    "New law bans social media for under-16s",
                ],
                "Social media ban",
            ),
        ]

        for headlines, expected in examples:
            with self.subTest(headlines=headlines):
                label = generate_topic_label(headlines)
                self.assertEqual(label, expected)
                self.assertLessEqual(len(label.split()), 7)
                self.assertNotIn("?", label)
                self.assertNotIn("vs.", label.casefold())

    def test_actual_grid_examples_get_english_event_labels(self):
        examples = [
            (
                [
                    "Le prix Nobel de physique 2026 est décerné au Belge Francis Halzen pour ses travaux sur les neutrinos",
                    "Nobelpreis für Physik geht an Neutrinoforscher Francis Halzen",
                    "Nobel per la Fisica a Francis Halzen per la scoperta dei neutrini cosmici",
                    "Premio Nobel de Física para Francis Halzen por sus descubrimientos sobre los neutrinos",
                ],
                "Physics Nobel for Francis Halzen",
            ),
            (
                [
                    "Premio Nobel de Medicina para Karl Deisseroth, Peter Hegemann y Georg Nagel por la técnica que permite controlar el cerebro con luz",
                    "Optogenetik: Medizin-Nobelpreis für den Lichtschalter im Gehirn",
                    "Nobelprijs voor de Geneeskunde voor optogenetica, een manier om hersencellen te besturen",
                ],
                "Medicine Nobel for Optogenetics",
            ),
            (
                [
                    "États-Unis : les étudiants de Cornell accusent leur université d'avoir couvert un viol",
                    "Cornell-Universität: Neue Details zu Vergewaltigungsvorwürfen erhöhen Druck auf US-Eliteuni",
                    "Melding van groepsverkrachting opent een breed debat over onbestraft seksueel geweld op Amerikaanse campussen",
                    "Studentin wirft Cornell-Uni lasches Vorgehen nach Gruppenvergewaltigung vor",
                ],
                "Cornell sexual assault allegations",
            ),
            (
                [
                    "Mobilisation lycéenne : des premiers cortèges en cours en France",
                    "Schülerdemos in Frankreich: Wütender Protest, Angst vor Eskalation",
                    "Demonstrationen: Was die Schüler in Frankreich fordern, betrifft auch Deutschland",
                    "Schulproteste in Frankreich: Warum sind die Oberschüler so unzufrieden?",
                    "Il terzo atto delle proteste studentesche in Francia",
                ],
                "French student protests",
            ),
            (
                [
                    "Tote nach Schiffsunglück im Schwarzen Meer - Drohnenangriff?",
                    "Schiff im Schwarzen Meer vor Bulgarien nach Drohnenangriffen gesunken",
                    "Vor Bulgarien: Drohnenangriff auf zwei Schiffe im Schwarzen Meer",
                    "Deux navires frappés par des drones en mer Noire au large de la Bulgarie",
                    "Atak dronów u wybrzeży Bułgarii. Jeden statek zatonął",
                ],
                "Drone attacks on ships in Black Sea",
            ),
            (
                [
                    "In pictures: Jeffrey Archer, prolific author, parliamentarian and prisoner dies aged 86",
                    "Bestsellerautor Jeffrey Archer mit 86 Jahren gestorben",
                    "L'auteur de best sellers Jeffrey Archer est décédé lundi",
                ],
                "Jeffrey Archer dies",
            ),
            (
                [
                    "La última función de Messi con la albiceleste: Argentina despide a su jugador récord",
                    "Messi se despide de Argentina: cuándo es y cómo ver el partido",
                    "Afscheid van Lionel Messi bij Argentinië in tien unieke momenten",
                    "Messi vor emotionalem Teamabschied: Wird unvergesslich",
                    "Messis letzter Auftritt: Argentinien verabschiedet seinen Magier",
                ],
                "Messi's Argentina farewell",
            ),
            (
                [
                    "Separatist party projected to win Quebec election, adding new test to Canada's unity",
                    "Quebec poised to elect separatists to power for first time in more than a decade",
                ],
                "Quebec separatists win election",
            ),
        ]

        for headlines, expected in examples:
            with self.subTest(headlines=headlines):
                label = generate_topic_label(headlines)
                self.assertEqual(label, expected)
                self.assertTrue(label.isascii())
                self.assertLessEqual(len(label.split()), 7)


if __name__ == "__main__":
    unittest.main()
