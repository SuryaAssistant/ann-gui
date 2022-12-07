import PySimpleGUI as sg
import training #from training.py


sg.theme('reddit')   # Add a touch of color

# Create two column
column_one = [
    [
        sg.Text("Data Preview"),
    ],
    [
        sg.Image('./export/heatmap.png', size=(1000,600))
    ],

]

column_two = [
    [
        sg.Button("EXIT"),
    ],
    [
        sg.Text(" ", key="-TRAININGRESULT-"),
    ],
    [
        sg.Text(" ", key="-SCORE-"),
    ],


]

# ----- Full layout -----
layout = [
    # ROW 1
    [
        sg.Column(column_one),
        sg.VSeperator(),
        sg.Column(column_two),
    ],

]

window = sg.Window("ANN Kelompok 8", layout)

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "EXIT" or event == sg.WIN_CLOSED:
        break

window.close()