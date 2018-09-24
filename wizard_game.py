import random
import time
import itertools
import copy


class ObservedWizardGame:
    def __init__(self, no_players):
        self.no_players = no_players
        self.players = []
        self.game_round = 1
        self.scoreboard = [[None] * int(60 / no_players) for i in range(no_players)]
        self.trump = None
        self.trick_estimations = []

    def connect_player(self, player):
        self.players.append(player)

    def has_started(self):
        return len(self.players) == self.no_players

    def is_players_turn(self, player):
        return player == self.next_player

    def run(self):
        assert self.has_started()
        next_beginning_player = 0
        for game_round in range(1, 1 + int(60 / self.no_players)):
            deck = Deck()
            deck.deal(self.players, game_round)  # Wait for dealer.
            if len(deck.cards) is not 0:            
                self.trump = deck.draw_card()
            else:
                self.trump = None
            if self.trump is not None and self.trump[1:] == '14':
                self.trump = self.players[next_beginning_player].determine_trump(self.trump)
            self.trick_estimations = []
            for p in range(len(self.players)):
                self.trick_estimations.append(self.players[(next_beginning_player + p) % self.no_players].estimate_tricks(self.trick_estimations, self.trump))
            trickround_scores = [0] * self.no_players
            next_player = next_beginning_player
            for i in range(game_round):
                table = []
                for p in range(len(self.players)):
                    active_player = (next_player + p) % self.no_players
                    played_card = self.players[active_player].play_card(table, self.trump)
                    #print('Player ' + str(active_player) + ' plays ' + played_card)
                    table.append(played_card)
                # Judge who won this trickround.
                next_player = (next_player + self.get_winning_index(table)) % self.no_players
                #print('Player ' + str(next_player) + ' won this trick!')
                trickround_scores[next_player] += 1
            print(self.trick_estimations, '   ', trickround_scores)
            next_beginning_player = (next_beginning_player + 1) % self.no_players
            # Evaluate if estimations were met.
            for p in range(len(self.players)):
                estimation_difference = abs(trickround_scores[p] - self.trick_estimations[p])
                if estimation_difference == 0:
                    self.scoreboard[p][game_round-1] = self.trick_estimations[p] * 10 + 20
                else:
                    self.scoreboard[p][game_round-1] = - estimation_difference * 10
                if game_round > 1:
                    self.scoreboard[p][game_round-1] += self.scoreboard[p][game_round-2]
            print(self.scoreboard)
            game_round += 1

    def get_winning_index(self, table):
        for i in range(len(table)):
            if '14' in table[i]:
                return i
        # No Wizard here.
        trump_color = None
        for i in range(len(table)):
            if self.trump is None:
                if table[i][1] is not '0':
                    trump_color = table[i][0]
            elif self.trump[0] in table[i][0]:
                    trump_color = self.trump[0]
        if trump_color is None:
            for i in range(len(table)):
                if table[i][1] is not '0':
                    trump_color = table[i][0]
        print('trump color is ' + str(trump_color), ' game trump:' + str(self.trump))
        if trump_color is None:
            return 0  # Case when only Jesters have been played.
        highest = None
        for i in range(len(table)):
            if trump_color in table[i] and (highest is None or self.get_card_number(table[i]) > self.get_card_number(table[highest])):
                highest = i
        return highest

    def get_card_number(self, card):
        return int(card[1:])


class Deck:
    def __init__(self):
        self.cards = list(itertools.chain(
            ['R' + str(i) for i in range(15)],
            ['G' + str(i) for i in range(15)],
            ['B' + str(i) for i in range(15)],
            ['Y' + str(i) for i in range(15)]))
        random.shuffle(self.cards)
    
    def deal(self, players, game_round):
        for p in players:
            p.receive_hand(self.cards[0:game_round])
            self.cards = self.cards[game_round:]

    def draw_card(self):
        return self.cards.pop(1)


class WizardPlayer:
    def __init__(self):
        self.hand = None

    def receive_hand(self, hand):
        self.hand = hand

    def estimate_tricks(self, others_estimations, trump):
        return random.randint(0, len(self.hand))

    def get_valid_indices(self, table):
        filtered_hand = [i for i in range(len(self.hand))]
        if any(map(lambda x: '14' in x, table)):  # Any card is permitted if a wizard was played before.
            return filtered_hand
        # No wizard on the table.
        filtered_table = list(filter(lambda x: x[1:] is not '0', table))
        if len(filtered_table) is 0:
            return filtered_hand
        servables = list(filter(lambda i: filtered_table[0][0] in self.hand[i] or self.hand[i][1:] in ['0','14'], filtered_hand))
        if len(servables) is not 0:
            return servables
        # Can't serve, just play anything.
        return filtered_hand
    
    def play_card(self, table, trump):
        # Filter only valid cards.
        valid_indices = self.get_valid_indices(table)
        return self.hand.pop(valid_indices[random.randint(0, len(valid_indices) - 1)])

    def determine_trump(self, trump):
        return None

    def print_hand(self):
        print(self.hand)




if __name__ == '__main__':
    random.seed(time.clock())
    no_players = 6
    game = ObservedWizardGame(no_players)
    for p in range(no_players):
        game.connect_player(WizardPlayer())
    game.run()
