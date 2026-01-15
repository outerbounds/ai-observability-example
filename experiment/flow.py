from metaflow import FlowSpec, step, IncludeFile, pypi, card
import io

class WildfireFlow(FlowSpec):

    wfdata = IncludeFile('wfdata', default='california-wildfire-data.csv', is_text=False)

    @pypi(packages={'duckdb': '1.4.3', 'pyarrow': '22.0.0'})
    @step
    def start(self):
        import duckdb
        import pyarrow as pa
        import pyarrow.csv as csv

        table = csv.read_csv(pa.BufferReader(self.wfdata))
        con = duckdb.connect()
        con.register("wildfires", table)
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    WildfireFlow()
