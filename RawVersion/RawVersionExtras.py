import pandas as pd

def f_process_user_sa(exp_data, row):
    user_sa = list()

    ##extract user data
    for dic ,key1 ,key2 ,indx in exp_data:
        ##extract current value
        value = exp_data.loc[exp_data.index[row], (dic ,key1 ,key2 ,indx)]

        ##value needs to be single ie don't want a single value series (for some reason sometimes we are getting series)
        if isinstance(value, pd.Series):
            value = value.squeeze()

        ##change indx to str so the following if statements work
        indx = str(indx)  # change to string because sometimes blank is read in as nan

        ##checks if both slice and key2 exists
        if not ('Unnamed' in indx  or 'nan' in indx):
            indices = tuple(slice(*(int(i) if i else None for i in part.strip().split(':'))) for part in indx.split(','))  # creats a slice object from a string - note slice objects are not inclusive ie to select the first number it should look like [0:1]
        else:
            indices = None
        if not ('Unnamed' in key2):
            pass
        else:
            key2 = None

        user_sa.append({
            "operation": dic,
            "key1": key1,
            "key2": key2,
            "indices": indices,
            "value": value,
        })
    return user_sa

