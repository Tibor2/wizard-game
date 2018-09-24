import unittest
from wizard_game import WizardPlayer



class TestWizardPlayer(unittest.TestCase):
    def test1(self):
        p1 = WizardPlayer()
        table = []
        hand = ['B3', 'Y9', 'R10', 'G2']
        p1.receive_hand(hand)
        self.assertTrue(len(p1.get_valid_indices(table)) == len(hand))
        
        table = ['Y12']
        valid_indices = p1.get_valid_indices(table)
        self.assertEqual(len(valid_indices), 1)
        self.assertEqual(hand[valid_indices[0]], 'Y9')
        
        table = ['B8', 'Y12']
        valid_indices = p1.get_valid_indices(table)
        self.assertEqual(len(valid_indices), 1)
        self.assertEqual(hand[valid_indices[0]], 'B3')

        hand.append('B0')
        valid_indices = p1.get_valid_indices(table)
        valid_hand = [hand[i] for i in valid_indices]
        self.assertEqual(len(valid_indices), 2)
        self.assertTrue('B0' in valid_hand)
        self.assertTrue('B3' in valid_hand)
        
        hand.append('Y14')
        valid_indices = p1.get_valid_indices(table)
        valid_hand = [hand[i] for i in valid_indices]
        self.assertEqual(len(valid_indices), 3)
        self.assertTrue('B0' in valid_hand)
        self.assertTrue('B3' in valid_hand)
        self.assertTrue('Y14' in valid_hand)

    def test2(self):
        p1 = WizardPlayer()
        table = ['R14']
        hand = ['B3', 'Y9', 'R10', 'G2']
        p1.receive_hand(hand)
        valid_indices = p1.get_valid_indices(table)
        valid_hand = [hand[i] for i in valid_indices]

        self.assertTrue(len(valid_hand) == len(hand))
        for card in valid_hand:
            self.assertTrue(card in hand)

if __name__ == '__main__':
    unittest.main()
