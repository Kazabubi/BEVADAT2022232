import pandas as pd
import numpy as np

#2.1
class NJCleaner:

    #2.2
    def __init__(self, csv_path : str) -> None:
        self.data = pd.read_csv(filepath_or_buffer=csv_path)

    #2.3
    def order_by_scheduled_time(self) -> pd.DataFrame:
        df = self.data.sort_values(by="scheduled_time")
        return df
    
    #2.4
    def drop_columns_and_nan(self) -> pd.DataFrame:
        df = self.data.copy()
        df.drop(["from", "to"], axis=1, inplace=True)
        df.dropna(axis=0, inplace=True)
        return df
    
    #2.5
    def convert_date_to_day(self):
        df = self.data.copy()
        df["day"] = pd.to_datetime(df["date"])
        df["day"] = df["day"].dt.day_name()
        df.drop(["date"], axis=1, inplace=True)
        return df
    
    #2.6
    def convert_scheduled_time_to_part_of_the_day(self):
        df = self.data.copy()

        df["scheduled_time"] = pd.to_datetime(df["scheduled_time"], format='%Y-%m-%d %H:%M:%S')

        ihelp = df.set_index("scheduled_time", drop=False)

        condition = [df["scheduled_time"].isin(ihelp.between_time('04:00','07:59').index),
                     df["scheduled_time"].isin(ihelp.between_time('08:00','11:59').index),
                     df["scheduled_time"].isin(ihelp.between_time('12:00','15:59').index),
                     df["scheduled_time"].isin(ihelp.between_time('16:00','19:59').index),
                     df["scheduled_time"].isin(ihelp.between_time('20:00','23:59').index),
                     df["scheduled_time"].isin(ihelp.between_time('00:00','03:59').index)]
        
        labels = ["early_morning", "morning","afternoon","evening","night","late_night"]

        df["part_of_the_day"] = np.select(condition, labels)

        df.drop(["scheduled_time"], axis=1, inplace=True)

        return df
    
    #2.7
    def convert_delay(self):
        df = self.data.copy()
        de = np.zeros((df.shape[0],1))
        da = df["delay_minutes"]
        de[da>5] = 1
        df["delay"] = de
        df["delay"] = df["delay"].astype("int")
        return df
    
    #2.8
    def drop_unnecessary_columns(self):
        df = self.data.copy()
        df.drop(["delay_minutes"], axis=1, inplace=True)
        df.drop(["actual_time"], axis=1, inplace=True)
        df.drop(["train_id"], axis=1, inplace=True)
        return df
    
    #2.9
    def save_first_60k(self, path : str) -> None:
        wr = self.data.iloc[:60000,:]
        wr.to_csv(path_or_buf=path, index=False)
        pass

    #2.10
    def prep_df(self, path = 'data/NJ.csv') -> None:

        self.data = self.order_by_scheduled_time()

        self.data = self.drop_columns_and_nan()

        self.data = self.convert_date_to_day()

        self.data = self.convert_scheduled_time_to_part_of_the_day()

        self.data = self.convert_delay()

        self.data = self.drop_unnecessary_columns()

        self.save_first_60k(path)

        pass
    

#njc = NJCleaner("C:/Users/venus/Desktop/TZ5MYT_BEVADAT2022232/HAZI/HAZI06/NJ Transit + Amtrak.csv")

#njc.prep_df("C:/Users/venus/Desktop/TZ5MYT_BEVADAT2022232/HAZI/HAZI06/res.csv")