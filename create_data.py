import sys
import pandas as pd
import numpy as np
import feather
import datetime
import time

import cbpro

start_date = datetime.date(2018, 2, 1)
end_date = datetime.date.today() + datetime.timedelta(days=1)

date_list = [start_date + datetime.timedelta(days=x) for x in range(0, (end_date-start_date).days, 10)]

public_client = cbpro.PublicClient()

output = pd.DataFrame()

for test_date in date_list:
    date1 = str(test_date)
    date2 = str(test_date + datetime.timedelta(days=10))

    df = pd.DataFrame(public_client.get_product_historic_rates('LTC-EUR',
                                                               granularity=3600,
                                                               start=date1,
                                                               end=date2),
                      columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.sort_values(by=['time'], ascending=True, inplace=True)
    df['returns'] = df['close'] / df['close'].shift(1) - 1
    output = pd.concat([output, df])
    time.sleep(2)

feather.write_dataframe(output, dest='./ltc_eur_data.feather')