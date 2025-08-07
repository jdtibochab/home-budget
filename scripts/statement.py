import pandas
import json
import os
import re
from datetime import datetime
from functools import lru_cache
from io import StringIO  # Import StringIO for handling in-memory text streams
from .helpers import last_day_of_month

pandas.options.mode.chained_assignment = None


def parse_datetime(d,data_format):
    try:
        return datetime.strptime(d, data_format)
    except:
        return datetime.strptime(d, '%Y-%m-%d')

# Caching data loading to prevent redundant reads
@lru_cache(maxsize=None)
def load_data():
    recurring = pandas.read_csv("./scripts/recurring.csv", comment="#")
    recurring['start'] = pandas.to_datetime(recurring['start'], format="%m/%d/%Y", errors='coerce')
    recurring['end'] = pandas.to_datetime(recurring['end'], format="%m/%d/%Y", errors='coerce')
    
    nonrecurring = pandas.read_csv("./scripts/nonrecurring.csv", comment="#")
    nonrecurring['date'] = pandas.to_datetime(nonrecurring['date'], format="%m/%d/%Y", errors='coerce')
    nonrecurring['description'] = nonrecurring['description'].fillna("")

    exchange_rate = {
        "MXN": pandas.read_csv("./scripts/USD_to_MXN.csv", comment="#"),
        "COP": pandas.read_csv("./scripts/USD_to_COP.csv", comment="#")
    }
    for k, v in exchange_rate.items():
        v["Date"] = pandas.to_datetime(v['Date'], format="%m/%d/%Y", errors='coerce')
        # Interpret numbers with comma delimiter for thousands as float
        try:
            v['Price'] = v['Price'].str.replace(',', '').astype(float)
        except:
            pass

    # Interpreters and corrections
    with open("./scripts/interpreter.json") as f:
        interpreter = json.load(f)
    
    with open("./scripts/account_types.json") as f:
        account_types = json.load(f)
    
    with open("./scripts/description_corrections.json") as f:
        description_corrections = json.load(f)
    
    with open("./scripts/type_corrections.json") as f:
        type_corrections = json.load(f)
    
    return recurring, nonrecurring, exchange_rate, interpreter, account_types, description_corrections, type_corrections

def reload_data():
    load_data.cache_clear()
    global recurring, nonrecurring, exchange_rate, interpreter, account_types, description_corrections, type_corrections
    recurring, nonrecurring, exchange_rate, interpreter, account_types, description_corrections, type_corrections = load_data()

def month_in_years(years,month):
    for y in years:
        if y in month:
            return True
    return False

recurring, nonrecurring, exchange_rate, interpreter, account_types, description_corrections, type_corrections = load_data()
directory = "./data/"

class Data():
    def __init__():
        self.statement = pandas.DataFrame()

    @property
    def data(self):
        return []

    @property
    def statement(self):
        return pandas.concat([i.statement for i in self.data],axis=0).sort_values("date")

    @property
    def id(self):
        return ""

    @property
    def summary(self):
        return self.statement.groupby("type")[["amount"]].sum().fillna(0.).sort_values("amount",ascending=False)

    def get(self,id):
        assert id in [i.id for i in self.data], "Data {} not found".format(id)
        return next(i for i in self.data if i.id == id)
        
class AccountStatement(Data):
    def __init__(self,path):
        self.path = path
        self.expressions = {k:v for k,v in interpreter.get(self.bank).get(self.type).items() if isinstance(v,dict)}
        self.interpreter = {v:k for k,v in interpreter.get(self.bank).get(self.type).items() if not isinstance(v,dict)}
        
    def __repr__(self):
        return "Data {}".format(self.id)

    @property
    def id(self):
        return self.account

    @property
    def info(self):
        return self.path.split("/")[-1].split("_")[0]
        
    @property
    def bank(self):
        return re.findall("[a-z]{1,}",self.info.lower())[0]
        
    @property
    def account(self):
        return str(re.findall("\d{1,}",self.info.lower())[0])

    @property
    def date_format(self):
        return interpreter.get(self.bank).get("date_format","%m/%d/%Y")

    @property
    def currency(self):
        return interpreter.get(self.bank).get("currency","USD")

    @property
    def date(self):
        return datetime.strptime(self.path.split("/")[-2], '%Y-%m')

    @property
    def exchange_rate(self):
        if self.currency == "USD":
            return 1
        df = exchange_rate.get(self.currency)
        df = df[(df["Date"]>=self.date) & (df["Date"]<=last_day_of_month(self.date))]
        if df.empty:
            # Return last available exchange rate if no data for the month
            return exchange_rate.get(self.currency)["Price"].values[-1]
        # Return last exchange rate for the month
        return df["Price"].values[-1]
        
    @property
    def type(self):
        for k,v in account_types.items():
            if self.account in v:
                return k

    def _load_and_fix_csv(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Detect and fix missing ",," at the end of the header
        if not lines[0].strip().endswith(",,"):
            lines[0] = lines[0].strip() + ",,\n"

        # Load the fixed CSV into pandas
        return pandas.read_csv(StringIO(''.join(lines)))  # Use StringIO here
    @property
    def statement(self):
        statement = self._load_and_fix_csv(self.path)
        # statement = pandas.read_csv(self.path)
        for k,d in self.expressions.items():
            for i,m in d.items():
                statement[i] *= m
            statement[k] = statement[d.keys()].fillna(0.).sum(axis=1)
        statement = statement[list(self.interpreter.keys())+list(self.expressions.keys())].rename(self.interpreter,axis=1)
        statement["account"] = account_types

        # assert(statement['date'].apply(lambda x: parse_datetime(x,self.date_format)), "Problem with {}".format(self.id))
        try:
            statement['date'] = statement['date'].apply(lambda x: parse_datetime(x,self.date_format))
        except:
            print("Issue with {}".format(self.id))
        statement["type"] = statement["type"].fillna("") if "type" in statement.columns else self.account
        statement["amount"] = statement["amount"] / self.exchange_rate
        if "balance" in statement.columns:
            statement["balance"] = statement["balance"] / self.exchange_rate
        for pattern,repl in type_corrections.items():
            statement.loc[statement["type"].str.contains(pattern,regex=True),"type"] = repl
        for pattern,repl in description_corrections.items():
            statement.loc[statement["description"].str.contains(pattern,regex=True),"type"] = repl
        return statement.sort_values("date")

    @property
    def balance(self):
        if self.type == "debit":
            return self.statement.sort_values("date",ascending=False)["balance"].values[0]
        if self.type == "credit":
            tmp = self.statement.sort_values("date",ascending=True)
            date_cut = tmp[tmp["description"].str.contains("Payment Thank You")]["date"]
            if date_cut.empty:
                # If no "Payment Thank You" date is found, return the sum of all amounts
                return min(tmp["amount"].sum(), 0)
            date_cut = date_cut.values[0]
            # Return the sum of amounts after the last "Payment Thank You" date
            return min(tmp[tmp["date"] > date_cut]["amount"].sum(), 0)
        return None

    @property
    def initial(self):
        if self.type != "debit":
            return None
        return self.statement["balance"].values[-1]

class FullStatement(Data):
    def __init__(self,date):
        self.date = date
        self.start = datetime.strptime("{}-01".format(self.date), '%Y-%m-%d')
        self.end = last_day_of_month(datetime.strptime("{}-28".format(self.date), '%Y-%m-%d'))
        self.directory = directory + self.date + "/"
        self._recurring = recurring
        self._nonrecurring = nonrecurring

    def __repr__(self):
        return "Statement {}".format(self.id)

    @property
    def id(self):
        return self.date
        
    @property
    def data(self):
        return [AccountStatement(self.directory+i) for i in sorted(os.listdir(self.directory)) if "csv" in i.lower() and "summary" not in i]

    @property
    def statement(self):
        lst = []
        for i in self.data:
            df = i.statement.copy()
            # df["amount"] = df["amount"] / i.exchange_rate
            # if "balance" in df.columns:
            #     df["balance"] = df["balance"] / i.exchange_rate
            lst.append(df)
        return pandas.concat(lst,axis=0)
        
    @property
    def summary(self):
        return self.statement.groupby("type")[["amount"]].sum().fillna(0.).sort_values("amount",ascending=False)

    @property
    def balances(self):
        return {i.id:i.balance for i in self.data}
    
    @property
    def balance(self):
        return sum([i.balance for i in self.data])

    @property
    def recurring(self):
        df = self._recurring[(self._recurring["start"] <= self.start) & (self._recurring["end"] >= self.end)]
        df["date"] = df["start"]
        df = df.drop("end",axis=1).drop("start",axis=1)
        return df
    
    @property
    def nonrecurring(self):
        return self._nonrecurring[(self._nonrecurring["date"] >= self.start) & (self._nonrecurring["date"] <= self.end)]

    @property
    def expected(self):
        return pandas.concat([self.recurring,self.nonrecurring]).groupby("type").sum(numeric_only=True).sort_values("amount",ascending=False).rename({"amount":"expected"},axis=1)

    @property
    def comparison(self):
        df = pandas.concat([self.expected,self.summary],axis=1).fillna(0.)
        df["delta"] = df["amount"] - df["expected"]
        return df
    
    def update(self):
        self._recurring,self._nonrecurring = load_data()

class Summary(Data):
    def __init__(self,years=["2024"]):
        self.years = years
        self.year = last_day_of_month(datetime.strptime(years[-1]+"-12", '%Y-%m'))
        self.folders = [i for i in sorted(os.listdir(directory)) if month_in_years(years,i)]
        self.first_date = datetime.strptime("{}".format(self.folders[0]), '%Y-%m')
        self.months = []
        for year in years:
            for month in range(12):
                date = datetime.strptime("{}-{}".format(year,month+1), '%Y-%m')
                if date < self.first_date:
                    continue
                self.months.append("{}-%.2d".format(year) % (month+1))
        # self.months = ["{}-%.2d".format(year) % (i+1) for i in range(12) if i+1>=self.first_month]

    def __repr__(self):
        return "Summary"

    @property
    def id(self):
        return "_".join(self.years)
    
    @property
    def data(self):
        return [FullStatement(i) for i in self.folders if "." not in i]

    @property
    def summary(self):
        df = pandas.concat([i.summary.rename({"amount":i.date},axis=1) for i in self.data],axis=1).fillna(0.).sort_index()
        df[df.abs()<0.01] = 0
        df = df.loc[df.mean(axis=1).sort_values(ascending=False).index]
        return df

    @property
    def balances(self):
        return pandas.DataFrame.from_dict({i.id:i.balances for i in self.data}).fillna(0.)

    @property
    def balance(self):
        return self.balances.sum()

    @property
    def expected_data(self):
        return [FullStatement(i) for i in self.months]

    @property
    def expected_summary(self):
        df = pandas.concat([i.expected.rename({"expected":i.date},axis=1) for i in self.expected_data],axis=1).fillna(0.)
        df[df.abs()<0.01] = 0
        df = df.loc[df.mean(axis=1).sort_values(ascending=False).index]
        return df

    @property
    def projected(self):
        return pandas.concat([self.summary,self.expected_summary[[i for i in self.expected_summary if i not in self.summary]]],axis=1).fillna(0.)

    @property
    def projected_balance(self):
        balance = self.balance.copy()
        for i in self.expected_data:
            if i.id in balance:
                continue
            balance[i.id] = balance.iloc[-1] + i.expected.sum().values[0]
        return balance

    @property
    def deltas(self):
        df = pandas.concat([i.comparison[["delta"]].rename({"delta":i.date},axis=1) for i in self.data]).fillna(0.)
        df = df.reset_index().groupby("type").sum(numeric_only=True)
        df = df[(df.abs()>0).any(axis=1)]
        df = df.loc[df.mean(axis=1).sort_values(ascending=True).index]
        return df
    
    def get_subscriptions(self):
        if len(self.data) <= 1:
            print("Not enough data to determine subscriptions.")
            return None

        # Get all rows whose description appears once very month
        lists = []
        for i in self.data:
            statement = i.statement
            spent = statement[(statement["amount"] < 0) & (~statement["type"].str.contains("Food|Groceries|Gas|SDGE|Internet|Shopping|Health"))]
            lists.append(spent["description"].unique().tolist())
        
        # Get instances that appear at least twice in all months
        from collections import defaultdict
        Counter = defaultdict(int)
        for lst in lists[1:]:
            for i in lst:
                Counter[i] += 1
        intersection = [k for k,v in Counter.items() if v >= 2]
        if not intersection:
            print("No subscriptions found.")
            return None

        # Get all rows that match the intersection
        subscriptions = self.data[-1].statement
        subscriptions = subscriptions[subscriptions["description"].isin(intersection)]
        return subscriptions[subscriptions["amount"] < 0].sort_values("date")
        
    def update(self):
        reload_data()

    def get(self,id):
        try:
            return next(i for i in self.data if i.id == id)
        except:
            return next(i for i in self.expected_data if i.id == id)
        
