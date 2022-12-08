import PySimpleGUI as sg
import training #from training.py


sg.theme('reddit')   # Add a touch of color

# Create two column
column_one = [
    [
        sg.Text("Hidden Layer Configuration"),
    ],
    [
        sg.Image('ann_resize.png', size=(300,300))
    ],
    [
        sg.Text("Hidden Layer Dimension"),
        sg.InputText(key="-HL-", do_not_clear=True, size=(10,1)),
    ],
    [
        sg.Text("Epoch"),
        sg.InputText(key="-EPOCH-", do_not_clear=True, size=(10,1)),
    ],
    [
        sg.Text("Train Size"),
        sg.InputText(key="-TRAINSIZE-", do_not_clear=True, size=(10,1)),
        sg.Text("%"),
    ],
]

column_two = [
    [
        sg.Button("SAVE"),sg.Button("EXIT"),
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

    if event == "SAVE" :

        window['-TRAININGRESULT-'].update("Training process ...")

        layer1 = int(values["-HL-"])
        ep = int(values["-EPOCH-"])
        tr_size = int(values["-TRAINSIZE-"])

        get_score = training.training_dataset(hidden_dimension = layer1, epoch=ep, train_size=tr_size)

        window['-TRAININGRESULT-'].update("ANN Model Updated.")
        text = "Score : " + str(get_score)
        window['-SCORE-'].update(text)

window.close()