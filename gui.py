import PySimpleGUI as sg
import os.path
import joblib

sg.theme('reddit')   # Add a touch of color

# Create two column
column_one = [
    [
        sg.Text("Fill data below"),
    ],
    [
        sg.Text("Age"),
        sg.InputText(key="-AGE-", do_not_clear=True, size=(10,1)),
    ],
    [
        sg.Text("Balance"),
        sg.InputText(key="-BALANCE-", do_not_clear=True, size=(10,1)),
    ],
    [
        sg.Text("Credit Score"),
        sg.InputText(key="-CREDIT-", do_not_clear=True, size=(10,1)),
    ],
    [
        sg.Text("Active Member?"),
        sg.Checkbox('Yes', default=False, key="-YES-"),
        sg.Checkbox('No', default=False, key="-NO-"),
    ],
    [
        sg.Button("PREDICT"), sg.Text("Result : ", key="-RESULT-"),
    ],
]

column_two = [
    [
        sg.Button("TRAINING"),
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

    # ROW 2
    [
        sg.Button("EXIT")
    ]
]

window = sg.Window("ANN Kelompok 8", layout)

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "EXIT" or event == sg.WIN_CLOSED:
        break
    
    if event == "TRAINING":
        from subprocess import call
        call(["python3", "training_gui.py"])


    if event == "PREDICT" :
        if values["-YES-"] == True:
            active = 1
        elif values["-NO-"] == True:
            active = 0
        
        age = values["-AGE-"]
        balance = values["-BALANCE-"]
        credit = values["-CREDIT-"]

        # Load model
        ann_model = joblib.load("model.joblib")
        to_predict = [0, 0, 0, 0]

        import numpy as np
        to_predict[0] = age
        to_predict[1] = balance
        to_predict[2] = active
        to_predict[3] = credit

        import pickle as pkl
        with open("scaler.pkl", "rb") as infile:
            scaler = pkl.load(infile)
            to_predict = np.array(to_predict)
            to_predict = to_predict.reshape(1, -1)
            to_predict = scaler.transform(to_predict)

        get_result = ann_model.predict(to_predict)

        if get_result == 1:
            window['-RESULT-'].update('Result : EXIT', background_color= "red", text_color="white")
        if get_result == 0:
            window['-RESULT-'].update('Result : STAY', background_color= "green", text_color="white")

window.close()