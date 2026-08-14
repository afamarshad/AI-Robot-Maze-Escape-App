import streamlit as st
import streamlit.components.v1 as components
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
    padding-top: 3.4rem !important;
    padding-bottom: 1rem !important;
    max-width: 1100px !important;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.35rem;
}

footer {
    visibility: hidden;
}

.stDeployButton {
    display: none;
}


/* ==========================================================
   TOP NAVBAR
   ========================================================== */

.top-navbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;

    height: 52px;

    display: flex;
    align-items: center;
    justify-content: center;

    background:
        linear-gradient(
            135deg,
            #0f172a,
            #172554
        );

    border-bottom: 1px solid #334155;

    box-shadow:
        0 3px 12px rgba(
            15,
            23,
            42,
            0.18
        );

    z-index: 999999;
}

.navbar-brand {

    color: #ffffff;

    font-size: 16px;

    font-weight: 750;

    letter-spacing: 0.2px;

    text-align: center;

    white-space: nowrap;
}


/* ==========================================================
   PAGE TITLE
   ========================================================== */

h1 {
    font-weight: 750 !important;
    letter-spacing: -0.5px !important;
}


/* ==========================================================
   TABS
   ========================================================== */

button[data-baseweb="tab"] {
    font-size: 17px !important;
    font-weight: 650 !important;
    padding-top: 8px !important;
    padding-bottom: 9px !important;
}


/* ==========================================================
   GENERAL BUTTONS
   ========================================================== */

.stButton > button {

    min-height: 43px !important;
    height: 43px !important;

    border-radius: 11px !important;

    border: 1px solid #d5dee9 !important;

    background: linear-gradient(
        145deg,
        #ffffff,
        #f1f5f9
    ) !important;

    color: #1e293b !important;

    font-size: 17px !important;

    font-weight: 650 !important;

    box-shadow:
        0 2px 5px rgba(
            15,
            23,
            42,
            0.08
        ) !important;

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease,
        border-color 0.15s ease !important;
}

.stButton > button:hover {

    border-color: #94a3b8 !important;

    box-shadow:
        0 5px 12px rgba(
            15,
            23,
            42,
            0.12
        ) !important;

    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(1px);
}


/* ==========================================================
   MOVEMENT BUTTONS
   ========================================================== */

.movement-button .stButton > button {

    min-height: 46px !important;

    height: 46px !important;

    font-size: 20px !important;

    border-radius: 12px !important;
}


/* ==========================================================
   CONTROL PANEL
   ========================================================== */

.control-title {

    font-size: 20px;

    font-weight: 750;

    color: #172033;

    line-height: 1.25;

    margin-bottom: 15px;
}

.control-subtitle {

    font-size: 11px;

    color: #64748b;

    line-height: 1.5;

    margin-bottom: 16px;
}

.section-label {

    font-size: 11px;

    font-weight: 750;

    color: #475569;

    letter-spacing: 0.7px;

    margin-top: 10px;

    margin-bottom: 11px;
}

.movement-section {

    margin-top: 14px;

    margin-bottom: 12px;
}

.game-actions-section {

    margin-top: 16px;

    margin-bottom: 12px;
}

.game-action-button {

    margin-top: 5px;

    margin-bottom: 8px;
}

.control-spacer {

    height: 85px;
}


/* ==========================================================
   VERTICAL SEPARATOR
   ========================================================== */

.vertical-separator {

    width: 2px;

    background:
        linear-gradient(
            to bottom,
            transparent,
            #cbd5e1 10%,
            #cbd5e1 90%,
            transparent
        );

    height: 425px;

    margin: 0 auto;
}


/* ==========================================================
   CONTAINERS
   ========================================================== */

div[data-testid="stVerticalBlockBorderWrapper"] {

    border-radius: 14px !important;

    border-color: #d9e2ec !important;

    background:
        linear-gradient(
            145deg,
            #ffffff,
            #f7fafc
        ) !important;

    box-shadow:
        0 4px 12px rgba(
            15,
            23,
            42,
            0.06
        ) !important;
}


/* ==========================================================
   INFORMATION CARDS
   ========================================================== */

.info-title {

    font-size: 11px;

    font-weight: 750;

    color: #475569;

    letter-spacing: 0.5px;

    margin-bottom: 1px;
}

.info-value {

    font-size: 24px;

    font-weight: 800;

    color: #172033;

    line-height: 1.15;

    margin-top: 2px;

    margin-bottom: 2px;
}


/* ==========================================================
   DIVIDERS
   ========================================================== */

hr {

    margin-top: 9px !important;

    margin-bottom: 10px !important;

    border-color: #dbe4ee !important;
}


/* ==========================================================
   ALERTS
   ========================================================== */

div[data-testid="stAlert"] {

    border-radius: 12px !important;
}


/* ==========================================================
   MOBILE
   ========================================================== */

@media (max-width: 700px) {

    .vertical-separator {
        display: none;
    }

    .control-spacer {
        height: 45px;
    }

    .top-navbar {
        height: 50px;
    }

    .navbar-brand {
        font-size: 14px;
    }

    .block-container {
        padding-top: 4.5rem !important;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TOP NAVBAR
# "Built By Afsah Arshad" MOVED HERE
# ============================================================

st.markdown("""
<div class="top-navbar">
    <div class="navbar-brand">
        Built By Afsah Arshad
    </div>
</div>
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

            if random.uniform(0, 1) < EPSILON:

                action = random.randint(0, 3)

            else:

                action = np.argmax(
                    q_table[current_state]
                )

            new_position = take_action(
                position,
                action
            )

            reward = get_reward(
                new_position
            )

            next_state = state_number(
                new_position
            )

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
# TRAIN AI
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


if "celebration_shown" not in st.session_state:

    st.session_state.celebration_shown = False


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

        st.session_state.celebration_shown = False


# ============================================================
# RESET GAME
# ============================================================

def reset_game():

    st.session_state.player_position = START

    st.session_state.moves = 0

    st.session_state.game_won = False

    st.session_state.show_ai_path = False

    st.session_state.celebration_shown = False


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "🤖 Robot Maze Escape"
)

st.caption(
    "Play the maze and see what the AI learned"
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
    # WIN MESSAGE + FULL-SCREEN BALLOON CELEBRATION
    # ========================================================

    if st.session_state.game_won:

        if not st.session_state.celebration_shown:

            st.balloons()

            st.session_state.celebration_shown = True

        st.success(
            f"🏆 Congratulations! "
            f"You reached the goal in "
            f"{st.session_state.moves} moves!"
        )


    # ========================================================
    # MAIN GAME AREA
    # CONTROLS SLIGHTLY WIDER / MAZE SLIGHTLY SMALLER
    # ========================================================

    control_column, separator_column, maze_column = st.columns(
        [1.25, 0.035, 2.7],
        gap="medium"
    )


    # ========================================================
    # CONTROL AREA
    # ========================================================

    with control_column:

        with st.container(border=True):

            # ------------------------------------------------
            # CONTROL TITLE
            # ------------------------------------------------

            st.markdown(
                '<div class="control-title">'
                '🎮 Controls'
                '</div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # CONTROL SUBTITLE
            # ------------------------------------------------

            st.markdown(
                '<div class="control-subtitle">'
                'Guide the robot through the maze'
                '</div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # EXTRA SPACE AFTER SUBTITLE
            # ------------------------------------------------

            st.markdown(
                '<div style="height:12px;"></div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # MOVEMENT LABEL
            # ------------------------------------------------

            st.markdown(
                '<div class="section-label">'
                'MOVEMENT'
                '</div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # UP BUTTON
            # ------------------------------------------------

            empty_left, up_col, empty_right = st.columns(
                [1, 1, 1],
                gap="small"
            )

            with up_col:

                if st.button(
                    "⬆️",
                    use_container_width=True,
                    key="up_button"
                ):

                    move_player(0)

                    st.rerun()


            # ------------------------------------------------
            # LEFT / DOWN / RIGHT
            # ------------------------------------------------

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


            # ------------------------------------------------
            # SEPARATOR
            # ------------------------------------------------

            st.markdown("---")


            # ------------------------------------------------
            # GAME ACTIONS LABEL
            # ------------------------------------------------

            st.markdown(
                '<div class="game-actions-section">'
                '<div class="section-label">'
                'GAME ACTIONS'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # SPACE BEFORE FIRST ACTION BUTTON
            # ------------------------------------------------

            st.markdown(
                '<div style="height:5px;"></div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # RESTART BUTTON
            # ------------------------------------------------

            if st.button(
                "🔄  Restart Maze",
                use_container_width=True,
                key="restart_button"
            ):

                reset_game()

                st.rerun()


            # ------------------------------------------------
            # SPACE BETWEEN ACTION BUTTONS
            # ------------------------------------------------

            st.markdown(
                '<div style="height:7px;"></div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # AI PATH BUTTON
            # ------------------------------------------------

            ai_button_text = (
                "🧠  Show AI Path"
                if not st.session_state.show_ai_path
                else "👁️  Hide AI Path"
            )


            if st.button(
                ai_button_text,
                use_container_width=True,
                key="show_ai_path_button"
            ):

                st.session_state.show_ai_path = (
                    not st.session_state.show_ai_path
                )

                st.rerun()


            # ------------------------------------------------
            # SPACE BEFORE INFORMATION
            # ------------------------------------------------

            st.markdown(
                '<div style="height:8px;"></div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # INSTRUCTIONS
            # ------------------------------------------------

            if not st.session_state.show_ai_path:

                st.info(
                    "💡 Reach 🏆 State 15 to win."
                )

                st.caption(
                    "The AI solution is hidden while you play."
                )

            else:

                st.success(
                    "🟢 Green cells show the AI's learned path."
                )

                st.caption(
                    "Dark-green borders highlight the AI path."
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
    # MAZE SIDE
    # ========================================================

    with maze_column:

        # ====================================================
        # INFORMATION CARDS
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
                    '<div class="info-title">'
                    '🤖 YOUR STATE'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="info-value">'
                    f'{state_number(st.session_state.player_position)}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    "Current position"
                )


        # ====================================================
        # YOUR MOVES
        # ====================================================

        with info2:

            with st.container(border=True):

                st.markdown(
                    '<div class="info-title">'
                    '👣 YOUR MOVES'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="info-value">'
                    f'{st.session_state.moves}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    "Moves taken"
                )


        # ====================================================
        # GOAL
        # ====================================================

        with info3:

            with st.container(border=True):

                st.markdown(
                    '<div class="info-title">'
                    '🏆 GOAL'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="info-value">'
                    '15'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    "Target state"
                )


        # ====================================================
        # SEPARATOR
        # ====================================================

        st.markdown("---")


        # ====================================================
        # CREATE MAZE CELLS
        # ====================================================

        maze_cells = ""


        for row in range(GRID_SIZE):

            for col in range(GRID_SIZE):

                position = (
                    row,
                    col
                )

                state = state_number(position)


                # =================================================
                # AI PATH MODE
                # =================================================

                if st.session_state.show_ai_path:

                    if position == GOAL:

                        maze_cells += f"""
                        <div class="maze-cell ai-goal">
                            <div class="icon">🏆</div>
                            <div class="cell-label">GOAL</div>
                            <div class="state">State {state}</div>
                        </div>
                        """

                    elif position == START:

                        maze_cells += f"""
                        <div class="maze-cell ai-start">
                            <div class="icon">🤖</div>
                            <div class="cell-label">START</div>
                            <div class="state">State {state}</div>
                        </div>
                        """

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

                        maze_cells += f"""
                        <div class="maze-cell ai-path">
                            <div class="arrow">{arrow}</div>
                            <div class="cell-label">AI PATH</div>
                            <div class="state">State {state}</div>
                        </div>
                        """

                    else:

                        maze_cells += f"""
                        <div class="maze-cell normal">
                            <div class="cell-label">STATE</div>
                            <div class="state">State {state}</div>
                        </div>
                        """

                # =================================================
                # NORMAL GAME MODE
                # =================================================

                else:

                    # ---------------------------------------------
                    # CURRENT PLAYER
                    # ---------------------------------------------

                    if (
                        position
                        ==
                        st.session_state.player_position
                    ):

                        # -----------------------------------------
                        # PLAYER REACHED GOAL
                        # SHOW ROBOT + POPPER ICON
                        # -----------------------------------------

                        if st.session_state.game_won:

                            maze_cells += f"""
                            <div class="maze-cell current">
                                <div class="icon">🤖 🎉</div>
                                <div class="cell-label">GOAL REACHED!</div>
                                <div class="state">State {state}</div>
                            </div>
                            """

                        else:

                            maze_cells += f"""
                            <div class="maze-cell current">
                                <div class="icon">🤖</div>
                                <div class="cell-label">YOU</div>
                                <div class="state">State {state}</div>
                            </div>
                            """

                    # ---------------------------------------------
                    # GOAL
                    # ---------------------------------------------

                    elif position == GOAL:

                        maze_cells += f"""
                        <div class="maze-cell goal">
                            <div class="icon">🏆</div>
                            <div class="cell-label">GOAL</div>
                            <div class="state">State {state}</div>
                        </div>
                        """

                    # ---------------------------------------------
                    # NORMAL
                    # ---------------------------------------------

                    else:

                        maze_cells += f"""
                        <div class="maze-cell normal">
                            <div class="cell-label">STATE</div>
                            <div class="state">State {state}</div>
                        </div>
                        """


        # ====================================================
        # MAZE HEADER TEXT
        # ====================================================

        if st.session_state.show_ai_path:

            maze_title = "🧠 AI LEARNED PATH"

            maze_subtitle = "Follow the green route"

        else:

            maze_title = "🧩 ROBOT MAZE"

            maze_subtitle = "State 0 → State 15"


        # ====================================================
        # MAZE HTML
        # ====================================================

        maze_html = f"""
<!DOCTYPE html>

<html>

<head>

<style>

* {{
    box-sizing: border-box;
}}

html,
body {{
    margin: 0;
    padding: 0;
    background: transparent;
    font-family: Arial, sans-serif;
}}

.maze-wrapper {{

    width: 100%;

    background:
        linear-gradient(
            145deg,
            #0f172a,
            #172554
        );

    border-radius: 22px;

    padding: 18px;

    border: 1px solid #334155;

    box-shadow:
        0 12px 30px rgba(
            15,
            23,
            42,
            0.20
        );

}}

.maze-header {{

    display: flex;

    justify-content: space-between;

    align-items: center;

    padding:
        0 4px 12px 4px;

}}

.maze-title {{

    color: white;

    font-size: 18px;

    font-weight: 700;

}}

.maze-subtitle {{

    color: #cbd5e1;

    font-size: 11px;

}}

.maze-grid {{

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 9px;

    background: #020617;

    padding: 10px;

    border-radius: 16px;

    border: 1px solid #334155;

}}

.maze-cell {{

    height: 78px;

    border-radius: 15px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    text-align: center;

    overflow: hidden;

}}

.normal {{

    background:
        linear-gradient(
            145deg,
            #1e3a5f,
            #27496d
        );

    color: #e2e8f0;

    border: 1px solid #42627f;

}}

.current {{

    background:
        linear-gradient(
            145deg,
            #dcfce7,
            #bbf7d0
        );

    color: #14532d;

    border: 3px solid #14532d;

    box-shadow:
        0 0 0 2px
        rgba(20,83,45,0.15),

        0 6px 16px
        rgba(22,101,52,0.30);

    font-weight: 700;

}}

.goal {{

    background:
        linear-gradient(
            145deg,
            #fef3c7,
            #fde68a
        );

    color: #92400e;

    border: 2px solid #f59e0b;

    font-weight: 700;

}}

.ai-start {{

    background:
        linear-gradient(
            145deg,
            #16a34a,
            #166534
        );

    color: white;

    border: 3px solid #052e16;

    box-shadow:
        0 5px 15px
        rgba(22,101,52,0.35);

    font-weight: 700;

}}

.ai-path {{

    background:
        linear-gradient(
            145deg,
            #22c55e,
            #15803d
        );

    color: white;

    border: 3px solid #14532d;

    box-shadow:
        0 0 0 2px
        rgba(20,83,45,0.15),

        0 5px 14px
        rgba(22,101,52,0.25);

    font-weight: 700;

}}

.ai-goal {{

    background:
        linear-gradient(
            145deg,
            #fbbf24,
            #d97706
        );

    color: white;

    border: 3px solid #14532d;

    box-shadow:
        0 5px 15px
        rgba(217,119,6,0.30);

    font-weight: 700;

}}

.icon {{

    font-size: 24px;

    line-height: 25px;

    margin-bottom: 3px;

}}

.cell-label {{

    font-size: 11px;

    font-weight: 700;

    line-height: 15px;

}}

.state {{

    font-size: 10px;

    opacity: 0.75;

    margin-top: 2px;

}}

.arrow {{

    font-size: 20px;

    font-weight: 800;

    line-height: 22px;

}}

.legend {{

    display: flex;

    justify-content: center;

    gap: 18px;

    padding-top: 12px;

    color: #cbd5e1;

    font-size: 11px;

}}

.legend-item {{

    display: flex;

    align-items: center;

    gap: 5px;

}}

.legend-box {{

    width: 10px;

    height: 10px;

    border-radius: 3px;

}}

.legend-normal {{
    background: #27496d;
}}

.legend-current {{

    background: #22c55e;

    border: 2px solid #14532d;

}}

.legend-goal {{
    background: #f59e0b;
}}

</style>

</head>

<body>

<div class="maze-wrapper">

    <div class="maze-header">

        <div class="maze-title">
            {maze_title}
        </div>

        <div class="maze-subtitle">
            {maze_subtitle}
        </div>

    </div>

    <div class="maze-grid">

        {maze_cells}

    </div>

    <div class="legend">

        <div class="legend-item">

            <span class="legend-box legend-normal"></span>

            Normal

        </div>

        <div class="legend-item">

            <span class="legend-box legend-current"></span>

            Current

        </div>

        <div class="legend-item">

            <span class="legend-box legend-goal"></span>

            Goal

        </div>

    </div>

</div>

</body>

</html>
"""


        # ====================================================
        # RENDER MAZE
        # ====================================================

        components.html(
            maze_html,
            height=425,
            scrolling=False
        )


# ============================================================
# AI Q-TABLE & MOVES TAB
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


    st.markdown("---")


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


        if position == GOAL:

            best_action = "🏆 Goal"

            best_value = 0

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


    prediction_df = pd.DataFrame(
        prediction_data,
        columns=[
            "State",
            "Position",
            "AI Best Action",
            "Q-Value"
        ]
    )


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