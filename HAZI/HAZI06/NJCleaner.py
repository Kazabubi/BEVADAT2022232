import pandas as pd
import numpy as np

#2.1
class NJCleaner:

    #2.2
    def __init__(self, csv_path : str) -> None:
        self.data = pd.read_csv(filepath_or_buffer=csv_path)

    #2.3
    def order_by_scheduled_time(self) -> pd.DataFrame:
        self.data.sort_values(by="scheduled_time", inplace=True)
        return self.data
    
    #2.4
    def drop_columns_and_nan(self) -> pd.DataFrame:
        self.data.drop(["from", "to"], axis=1, inplace=True)
        self.data.dropna(axis=0, inplace=True)
        return self.data
    
    #2.5
    def convert_date_to_day(self):
        self.data["day"] = pd.to_datetime(self.data["date"])
        self.data["day"] = self.data["day"].dt.day_name()
        self.data.drop(["date"], axis=1, inplace=True)
        return self.data
    
    #2.6
    def convert_scheduled_time_to_part_of_the_day(self):
        self.data["scheduled_time"] = pd.to_datetime(self.data["scheduled_time"], format='%Y-%m-%d %H:%M:%S')

        ihelp = self.data.set_index("scheduled_time", drop=False)

        condition = [self.data["scheduled_time"].isin(ihelp.between_time('04:00','07:59').index),
                     self.data["scheduled_time"].isin(ihelp.between_time('08:00','11:59').index),
                     self.data["scheduled_time"].isin(ihelp.between_time('12:00','15:59').index),
                     self.data["scheduled_time"].isin(ihelp.between_time('16:00','19:59').index),
                     self.data["scheduled_time"].isin(ihelp.between_time('20:00','23:59').index),
                     self.data["scheduled_time"].isin(ihelp.between_time('00:00','03:59').index)]
        
        labels = ["early_morning", "morning","afternoon","evening","night","late_night"]

        self.data["part_of_the_day"] = np.select(condition, labels)

        self.data.drop(["scheduled_time"], axis=1, inplace=True)

        return self.data
    
    #2.7
    def convert_delay(self):
        de = np.zeros((self.data.shape[0],1))
        da = self.data["delay_minutes"]
        de[da>5] = 1
        self.data["delay"] = de
        return self.data
    
    #2.8
    def drop_unnecessary_columns(self):
        self.data.drop(["delay_minutes"], axis=1, inplace=True)
        self.data.drop(["actual_time"], axis=1, inplace=True)
        self.data.drop(["train_id"], axis=1, inplace=True)
        return self.data
    
    #2.9
    def save_first_60k(self, csv_path : str) -> None:
        wr = self.data.iloc[:60000,:]
        wr.to_csv(path_or_buf=csv_path)
        pass

    #2.10
    def prep_df(self, csv_path = 'data/NJ.csv') -> None:

        self.order_by_scheduled_time()

        self.drop_columns_and_nan()

        self.convert_date_to_day()

        self.convert_scheduled_time_to_part_of_the_day()

        self.convert_delay()

        self.drop_unnecessary_columns()

        self.save_first_60k(csv_path)

        pass
    

njc = NJCleaner("C:/Users/venus/Desktop/TZ5MYT_BEVADAT2022232/HAZI/HAZI06/NJ Transit + Amtrak.csv")

njc.prep_df("C:/Users/venus/Desktop/TZ5MYT_BEVADAT2022232/HAZI/HAZI06/res.csv")