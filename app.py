import streamlit as st
import numpy as np
import random
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Robot Maze Escape",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 3rem !important;
    padding-bottom: 0.5rem !important;
    max-width: 1100px !important;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.25rem;
}


/* ============================================================
   TABS
   ============================================================ */

button[data-baseweb="tab"] {
    font-size: 18px !important;
    font-weight: 200 !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    min-height: 42px !important;
    height: 42px !important;
    border-radius: 10px !important;
    font-size: 19px !important;
}


/* ============================================================
   MAIN VERTICAL SEPARATOR
   ============================================================ */

.vertical-separator {
    width: 1px;
    background-color: #e2e8f0;
    height: 100%;
    min-height: 420px;
    margin: 0 auto;
}


/* ============================================================
   BOTTOM SEPARATOR
   ============================================================ */

.bottom-separator {
    width: 100%;
    height: 1px;
    background-color: #e2e8f0;
    margin-top: 2px;
    margin-bottom: 8px;
}


/* ============================================================
   INFO BOXES
   ============================================================ */

.info-label {
    font-size: 14px;
    font-weight: 600;
    color: #475569;
    text-align: center;
    margin-bottom: 0px;
}

.info-value {
    font-size: 24px;
    font-weight: 400;
    color: #334155;
    text-align: center;
    margin-top: 0px;
}


/* ============================================================
   MAZE CELLS
   ============================================================ */

.maze-cell {
    width: 100%;
    height: 42px;
    min-height: 42px;
    box-sizing: border-box;

    border-radius: 10px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 15px;
    font-weight: 600;

    margin: 0;
    padding: 0;
}


/* ============================================================
   NORMAL CELL
   ============================================================ */

.normal-maze-cell {
    background-color: #e0f2fe;
    color: #334155;
    border: 1px solid #cbd5e1;
}


/* ============================================================
   CURRENT PLAYER CELL
   ============================================================ */

/*
   IMPORTANT:
   This is the active/current state.

   Dark green 2px border.
*/

.current-maze-cell {
    background-color: #dcfce7;
    color: #166534;

    border: 2px solid #14532d !important;

    font-weight: 700;
}


/* ============================================================
   AI PATH CELL
   ============================================================ */

/*
   Every cell that belongs to the AI learned path
   receives the same dark green 2px border.
*/

.ai-path-maze-cell {
    background-color: #22c55e;
    color: white;

    border: 2px solid #14532d !important;

    font-weight: 700;
}


/* ============================================================
   AI START CELL
   ============================================================ */

.ai-start-maze-cell {
    background-color: #16a34a;
    color: white;

    border: 2px solid #14532d !important;

    font-weight: 700;
}


/* ============================================================
   AI GOAL CELL
   ============================================================ */

.ai-goal-maze-cell {
    background-color: #f59e0b;
    color: white;

    border: 2px solid #14532d !important;

    font-weight: 700;
}


/* ============================================================
   NORMAL GOAL CELL
   ============================================================ */

.goal-maze-cell {
    background-color: #fef3c7;
    color: #92400e;

    border: 1px solid #f59e0b;

    font-weight: 700;
}


/* ============================================================
   HIDE STREAMLIT FOOTER
   ============================================================ */

footer {
    visibility: hidden;
}

.stDeployButton {
    display: none;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# Q-LEARNING SETTINGS
# ============================================================

GRID_SIZE = 4

START = (0, 0)

GOAL = (3, 3)


# ============================================================
# ACTIONS
# ============================================================

# 0 = Up
# 1 = Down
# 2 = Left
# 3 = Right

ACTIONS = [
    "↑",
    "↓",
    "←",
    "→"
]

ACTION_NAMES = [
    "Up",
    "Down",
    "Left",
    "Right"
]


# ============================================================
# Q-LEARNING PARAMETERS
# ============================================================

ALPHA = 0.1

GAMMA = 0.9

EPSILON = 0.2

EPISODES = 500


# ============================================================
# CONVERT POSITION TO STATE NUMBER
# ============================================================

def state_number(position):

    row, col = position

    return row * GRID_SIZE + col


# ============================================================
# TAKE ACTION
# ============================================================

def take_action(position, action):

    row, col = position

    if action == 0:
        row -= 1

    elif action == 1:
        row += 1

    elif action == 2:
        col -= 1

    elif action == 3:
        col += 1

    row = max(
        0,
        min(row, GRID_SIZE - 1)
    )

    col = max(
        0,
        min(col, GRID_SIZE - 1)
    )

    return row, col


# ============================================================
# REWARD FUNCTION
# ============================================================

def get_reward(new_position):

    if new_position == GOAL:
        return 10

    return -1


# ============================================================
# TRAIN Q-LEARNING
# ============================================================

@st.cache_resource
def train_q_learning():

    random.seed(42)
    np.random.seed(42)

    q_table = np.zeros(
        (
            GRID_SIZE * GRID_SIZE,
            4
        )
    )

    for episode in range(EPISODES):

        position = START

        for step in range(100):

            current_state = state_number(position)

            # Epsilon-greedy action selection
            if random.uniform(0, 1) < EPSILON:

                action = random.randint(0, 3)

            else:

                action = np.argmax(
                    q_table[current_state]
                )

            # Take action
            new_position = take_action(
                position,
                action
            )

            # Get reward
            reward = get_reward(
                new_position
            )

            # Get next state
            next_state = state_number(
                new_position
            )

            # Q-learning update
            q_table[
                current_state,
                action
            ] = (
                q_table[
                    current_state,
                    action
                ]
                +
                ALPHA
                *
                (
                    reward
                    +
                    GAMMA
                    *
                    np.max(
                        q_table[next_state]
                    )
                    -
                    q_table[
                        current_state,
                        action
                    ]
                )
            )

            position = new_position

            if position == GOAL:
                break

    return q_table


# ============================================================
# GET AI LEARNED PATH
# ============================================================

def get_ai_path(q_table):

    position = START

    path = [position]

    visited = set()

    for step in range(20):

        if position == GOAL:
            break

        if position in visited:
            break

        visited.add(position)

        state = state_number(position)

        action = np.argmax(
            q_table[state]
        )

        new_position = take_action(
            position,
            action
        )

        path.append(
            new_position
        )

        position = new_position

        if position == GOAL:
            break

    return path


# ============================================================
# GET ARROW BETWEEN TWO POSITIONS
# ============================================================

def get_arrow(current, next_position):

    current_row, current_col = current

    next_row, next_col = next_position

    if next_row > current_row:
        return "↓"

    elif next_row < current_row:
        return "↑"

    elif next_col > current_col:
        return "→"

    elif next_col < current_col:
        return "←"

    return ""


# ============================================================
# CREATE MAZE CELL
# ============================================================

def render_maze_cell(
    position,
    cell_class,
    text
):

    st.html(
        f"""
        <div class="maze-cell {cell_class}">
            {text}
        </div>
        """
    )


# ============================================================
# TRAIN THE AI
# ============================================================

q_table = train_q_learning()


# ============================================================
# GET AI PATH
# ============================================================

ai_path = get_ai_path(q_table)


# ============================================================
# SESSION STATE
# ============================================================

if "player_position" not in st.session_state:
    st.session_state.player_position = START

if "moves" not in st.session_state:
    st.session_state.moves = 0

if "game_won" not in st.session_state:
    st.session_state.game_won = False

if "show_ai_path" not in st.session_state:
    st.session_state.show_ai_path = False


# ============================================================
# MOVE PLAYER
# ============================================================

def move_player(action):

    if st.session_state.game_won:
        return

    current_position = (
        st.session_state.player_position
    )

    new_position = take_action(
        current_position,
        action
    )

    if new_position != current_position:
        st.session_state.moves += 1

    st.session_state.player_position = new_position

    if new_position == GOAL:
        st.session_state.game_won = True


# ============================================================
# RESET GAME
# ============================================================

def reset_game():

    st.session_state.player_position = START

    st.session_state.moves = 0

    st.session_state.game_won = False

    st.session_state.show_ai_path = False


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🤖 Robot Maze Escape")

st.caption(
    "Play the maze and see what the AI learned"
)

st.caption(
    "Built By Afsah Arshad"
)


# ============================================================
# TABS
# ============================================================

game_tab, ai_tab = st.tabs(
    [
        "🎮 Play Maze",
        "🧠 AI Q-Table & Moves"
    ]
)


# ============================================================
# PLAY MAZE TAB
# ============================================================

with game_tab:

    # ========================================================
    # WIN MESSAGE
    # ========================================================

    if st.session_state.game_won:

        st.success(
            f"🏆 Congratulations! "
            f"You reached the goal in "
            f"{st.session_state.moves} moves!"
        )


    # ========================================================
    # MAIN GAME AREA
    # ========================================================

    control_column, separator_column, maze_column = st.columns(
        [1, 0.035, 3],
        gap="medium"
    )


    # ========================================================
    # LEFT SIDE - CONTROLS
    # ========================================================

    with control_column:

        st.markdown(
            "### 🎮 Controls"
        )

        st.caption(
            "Move the robot"
        )

        st.write("")


        # ----------------------------------------------------
        # UP
        # ----------------------------------------------------

        if st.button(
            "⬆️",
            use_container_width=True,
            key="up_button"
        ):

            move_player(0)

            st.rerun()


        # ----------------------------------------------------
        # LEFT / DOWN / RIGHT
        # ----------------------------------------------------

        left_col, down_col, right_col = st.columns(
            3,
            gap="small"
        )


        with left_col:

            if st.button(
                "⬅️",
                use_container_width=True,
                key="left_button"
            ):

                move_player(2)

                st.rerun()


        with down_col:

            if st.button(
                "⬇️",
                use_container_width=True,
                key="down_button"
            ):

                move_player(1)

                st.rerun()


        with right_col:

            if st.button(
                "➡️",
                use_container_width=True,
                key="right_button"
            ):

                move_player(3)

                st.rerun()


        st.write("")


        # ----------------------------------------------------
        # RESTART
        # ----------------------------------------------------

        if st.button(
            "🔄 Restart",
            use_container_width=True,
            key="restart_button"
        ):

            reset_game()

            st.rerun()


        # ----------------------------------------------------
        # SHOW / HIDE AI PATH
        # ----------------------------------------------------

        if st.button(
            "🧠 Show AI Path"
            if not st.session_state.show_ai_path
            else "🙈 Hide AI Path",
            use_container_width=True,
            key="show_ai_path_button"
        ):

            st.session_state.show_ai_path = (
                not st.session_state.show_ai_path
            )

            st.rerun()


        st.write("")


        # ----------------------------------------------------
        # INSTRUCTIONS
        # ----------------------------------------------------

        st.caption(
            "💡 Reach the 🏆 goal."
        )

        if not st.session_state.show_ai_path:

            st.caption(
                "The AI solution is hidden "
                "while you play."
            )

            st.caption(
                "Click 🧠 Show AI Path to "
                "see the learned route."
            )

        else:

            st.caption(
                "🟢 Green boxes show the "
                "AI's learned path."
            )

            st.caption(
                "The dark-green border "
                "highlights every path cell."
            )


    # ========================================================
    # VERTICAL SEPARATOR
    # ========================================================

    with separator_column:

        st.markdown(
            '<div class="vertical-separator"></div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # RIGHT SIDE - INFORMATION + MAZE
    # ========================================================

    with maze_column:

        # ====================================================
        # GAME INFORMATION BOXES
        # ====================================================

        info1, info2, info3 = st.columns(
            [1, 1, 1],
            gap="small"
        )


        # ====================================================
        # YOUR STATE
        # ====================================================

        with info1:

            with st.container(border=True):

                st.markdown(
                    "🤖 **YOUR STATE**"
                )

                st.markdown(
                    f"{state_number(st.session_state.player_position)}"
                )


        # ====================================================
        # YOUR MOVES
        # ====================================================

        with info2:

            with st.container(border=True):

                st.markdown(
                    "👣 **YOUR MOVES**"
                )

                st.markdown(
                    f"{st.session_state.moves}"
                )


        # ====================================================
        # GOAL
        # ====================================================

        with info3:

            with st.container(border=True):

                st.markdown(
                    "🏆 **GOAL**"
                )

                st.markdown(
                    "State 15"
                )


        # ====================================================
        # BOTTOM SEPARATOR
        # ====================================================

        st.markdown(
            '<div class="bottom-separator"></div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # MAZE TITLE
        # ====================================================

        if st.session_state.show_ai_path:

            st.markdown(
                "### 🧩 Maze — 🟢 AI Learned Path"
            )

        else:

            st.markdown(
                "### 🧩 Maze"
            )


        # ====================================================
        # CREATE 4 × 4 MAZE
        # ====================================================

        for row in range(GRID_SIZE):

            maze_columns = st.columns(
                GRID_SIZE,
                gap="small"
            )

            for col in range(GRID_SIZE):

                position = (
                    row,
                    col
                )

                with maze_columns[col]:

                    # ========================================
                    # AI PATH MODE
                    # ========================================

                    if st.session_state.show_ai_path:

                        # ------------------------------------
                        # GOAL
                        # ------------------------------------

                        if position == GOAL:

                            render_maze_cell(
                                position,
                                "ai-goal-maze-cell",
                                f"🏆 State {state_number(position)}"
                            )


                        # ------------------------------------
                        # START
                        # ------------------------------------

                        elif position == START:

                            render_maze_cell(
                                position,
                                "ai-start-maze-cell",
                                f"🤖 State {state_number(position)}"
                            )


                        # ------------------------------------
                        # AI PATH
                        # ------------------------------------

                        elif position in ai_path:

                            path_index = ai_path.index(
                                position
                            )

                            arrow = ""

                            if path_index < len(ai_path) - 1:

                                arrow = get_arrow(
                                    ai_path[path_index],
                                    ai_path[path_index + 1]
                                )

                            render_maze_cell(
                                position,
                                "ai-path-maze-cell",
                                f"🟢 {arrow} State "
                                f"{state_number(position)}"
                            )


                        # ------------------------------------
                        # NORMAL CELL
                        # ------------------------------------

                        else:

                            render_maze_cell(
                                position,
                                "normal-maze-cell",
                                f"State "
                                f"{state_number(position)}"
                            )


                    # ========================================
                    # NORMAL GAME MODE
                    # ========================================

                    else:

                        # ------------------------------------
                        # CURRENT PLAYER STATE
                        # ------------------------------------

                        if (
                            position
                            ==
                            st.session_state.player_position
                        ):

                            render_maze_cell(
                                position,
                                "current-maze-cell",
                                f"🤖 State "
                                f"{state_number(position)}"
                            )


                        # ------------------------------------
                        # GOAL
                        # ------------------------------------

                        elif position == GOAL:

                            render_maze_cell(
                                position,
                                "goal-maze-cell",
                                f"🏆 State "
                                f"{state_number(position)}"
                            )


                        # ------------------------------------
                        # NORMAL CELL
                        # ------------------------------------

                        else:

                            render_maze_cell(
                                position,
                                "normal-maze-cell",
                                f"State "
                                f"{state_number(position)}"
                            )


# ============================================================
# COMBINED AI Q-TABLE & MOVES TAB
# ============================================================

with ai_tab:

    st.title(
        "🧠 AI Q-Table & Moves"
    )

    st.write(
        "View the Q-values learned by the AI, "
        "the learned route, and the AI's decision "
        "for each state."
    )


    # ========================================================
    # AI Q-TABLE
    # ========================================================

    st.subheader(
        "📊 AI Q-Table"
    )

    st.write(
        "This table contains the values learned "
        "by the Q-Learning algorithm."
    )


    # ========================================================
    # CREATE Q-TABLE DATAFRAME
    # ========================================================

    qtable_df = pd.DataFrame(
        np.round(
            q_table,
            2
        ),
        columns=[
            "↑ Up",
            "↓ Down",
            "← Left",
            "→ Right"
        ]
    )

    qtable_df.insert(
        0,
        "State",
        range(16)
    )


    # ========================================================
    # DISPLAY Q-TABLE
    # ========================================================

    st.dataframe(
        qtable_df,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "For each state, the AI selects the action "
        "with the highest Q-value."
    )


    # ========================================================
    # Q-LEARNING PARAMETERS
    # ========================================================

    st.subheader(
        "⚙️ Q-Learning Parameters"
    )

    parameter_col1, parameter_col2 = st.columns(2)


    with parameter_col1:

        st.write(
            f"**Learning Rate (α):** {ALPHA}"
        )

        st.write(
            f"**Discount Factor (γ):** {GAMMA}"
        )


    with parameter_col2:

        st.write(
            f"**Exploration Probability (ε):** {EPSILON}"
        )

        st.write(
            f"**Training Episodes:** {EPISODES}"
        )


    # ========================================================
    # SEPARATOR
    # ========================================================

    st.markdown(
        '<div class="bottom-separator"></div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # AI LEARNED MOVES
    # ========================================================

    st.subheader(
        "🤖 AI's Best Route"
    )

    st.write(
        "Here you can see the route learned "
        "by the Q-Learning agent."
    )

    route = " → ".join(
        [
            f"S{state_number(position)}"
            for position in ai_path
        ]
    )

    st.success(
        route
    )

    st.write(
        f"**Total AI moves:** "
        f"{len(ai_path) - 1}"
    )


    # ========================================================
    # AI DECISION TABLE
    # ========================================================

    st.subheader(
        "🎯 AI Decision for Each State"
    )

    prediction_data = []


    for state in range(16):

        row = state // GRID_SIZE

        col = state % GRID_SIZE

        position = (
            row,
            col
        )


        # ----------------------------------------------------
        # GOAL STATE
        # ----------------------------------------------------

        if position == GOAL:

            best_action = "🏆 Goal"

            best_value = 0


        # ----------------------------------------------------
        # NORMAL STATE
        # ----------------------------------------------------

        else:

            action_index = np.argmax(
                q_table[state]
            )

            best_action = (
                f"{ACTIONS[action_index]} "
                f"{ACTION_NAMES[action_index]}"
            )

            best_value = q_table[
                state,
                action_index
            ]


        prediction_data.append(
            [
                state,
                position,
                best_action,
                round(
                    best_value,
                    2
                )
            ]
        )


    # ========================================================
    # CREATE DECISION DATAFRAME
    # ========================================================

    prediction_df = pd.DataFrame(
        prediction_data,
        columns=[
            "State",
            "Position",
            "AI Best Action",
            "Q-Value"
        ]
    )


    # ========================================================
    # DISPLAY DECISION TABLE
    # ========================================================

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    st.success(
        "🧠 The green path in the Play Maze tab "
        "shows the route selected by the trained "
        "Q-Learning agent."
    )