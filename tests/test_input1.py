from serpent.input_controller import KeyboardKey, KeyboardEvent, KeyboardEvents
from serpent.input_controller import MouseButton, MouseEvent, MouseEvents
from serpent.enums import InputControlTypes
game_inputs = {
    "MOVEMENT": {
    	"WALK LEFT": [
                    KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_A)
        ],
        "STRAFEALEFT": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_A),
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W),
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_LEFT_SHIFT)
       	],
        "SPRINT": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W),
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_LEFT_SHIFT)
        ],
        "WALK": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W)
        ],
        "STRAFE RIGHT": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W),
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_D),
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_LEFT_SHIFT)
        ],
        "WALK RIGHT": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_D)
        ],
        "BACK": [
        	KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_S)
        ],
        "JUMP": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_SPACE)
        ],
        "CROUCH": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_C)
        ],
        "PHRONE": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_LEFT_CTRL)
        ],
            "STOPPED": []
    },
    "COMBAT": {
        "RELOAD": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_R)
        ],
        "ABILTY": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_B)
        ],
        "GRENADE": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_G)
        ],
        "MELEE": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_E)
        ],
        "USE": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_F)
        ],
        "NON LETHAL": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_Q)
        ],
        "SECONDARY": [
            KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_1)
        ],
        "Neutral": []
            },
    "CURSOR": {
        "MOVE MOUSE X1": [],
        "MOVE MOUSE Y1": [],
        "MOVE MOUSE XY1": [],
        "MOVE MOUSE X2": [],
        "MOVE MOUSE Y2": [], 
        "MOVE MOUSE XY2": [],
        "MOVE MOUSE XY3": [],
        "MOVE MOUSE XY4": [],
        "IDLE_MOUSE": []
    },
    "FIRE": {
        "SHOOT1": [],
        "SHOOT2": [],
        "SHOOT3": [],
        "SHOOT4": [],
        "IDLE_FIRE": []
    }
}
def combine_game_inputs(self, combination):
        """ Combine game input axes in a single flattened collection
        Args:
        combination [list] -- A combination of valid game input axis keys
        """

        # Validation
        if not isinstance(combination, list):
            raise SerpentError("'combination' needs to be a list")

        for entry in combination:
            if isinstance(entry, list):
                for entry_item in entry:
                    if entry_item not in self.game_inputs:
                        raise SerpentError("'combination' entries need to be valid members of self.game_input...")
            else:
                if entry not in self.game_inputs:
                    raise SerpentError("'combination' entries need to be valid members of self.game_input...")

        # Concatenate Grouped Axes (if needed)
        game_input_axes = list()

        for entry in combination:
            if isinstance(entry, str):
                game_input_axes.append(self.game_inputs[entry])
            elif isinstance(entry, list):
                concatenated_game_input_axis = dict()

                for entry_item in entry:
                    concatenated_game_input_axis = {**concatenated_game_input_axis, **self.game_inputs[entry_item]}

                game_input_axes.append(concatenated_game_input_axis)

        # Combine Game Inputs
        game_inputs = dict()

        if not len(game_input_axes):
            return game_inputs

        for keys in itertools.product(*game_input_axes):
            compound_label = list()
            game_input = list()

            for index, key in enumerate(keys):
                compound_label.append(key)
                game_input += game_input_axes[index][key]

            game_inputs[" - ".join(compound_label)] = game_input

        return game_inputs
game_inputs = [
	{
    "name": "CONTROLS",
    "control_type": InputControlTypes.DISCRETE,
    "inputs": combine_game_inputs(game_inputs(["MOVEMENT", "COMBAT", "CURSOR"]))
    }
]
print(len(game_inputs[0]["inputs"]))