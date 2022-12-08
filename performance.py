import PySimpleGUI as sg
import training #from training.py


sg.theme('reddit')   # Add a touch of color

# Create two column
column_one = [
    [
        sg.Text("Data Preview"),
    ],
    [
        sg.Image('./export/accuracy.png', size=(600,600), key="-IMAGE-")
    ],

]

column_two = [
    [
        sg.Button("ACCURACY"),
    ],
    [
        sg.Button("LOSS"),
    ],
    [
        sg.Button("HEATMAP"),
    ],
    [
        sg.Button("BACK"),
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
    if event == "BACK" or event == sg.WIN_CLOSED:
        break
    if event == "ACCURACY" :
        window['-IMAGE-'].update('./export/accuracy.png')
    if event == "LOSS" :
        window['-IMAGE-'].update('./export/loss.png')
    if event == "HEATMAP" :
        window['-IMAGE-'].update('./export/heatmap.png')
window.close()