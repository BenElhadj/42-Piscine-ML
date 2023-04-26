import math


class TinyStatistician:

    def DataChecker(func):
        def wrapper(self, *args, **kwargs):
            data = args[0]
            if (len(data) != 1 and isinstance(data[0], list)\
                and len(data[0]) != 1):
                raise BaseException('SizeError: size of Vector !!')
            for item in data:
                if not isinstance(item, int) and not isinstance(item, float):
                    raise BaseException(
                        f"TypeError: {item} is not type float or dtype int)")
            data = list(filter(lambda x:  x == x, data))
            args = (data, args[1]) if len(args) > 1 else (data,)
            res = func(self, *args, **kwargs)
            if res is None:
                return None

            return round(float(res), 2) if func.__name__ != 'quartile'\
                else [float(res[0]), float(res[1])]
            # return float(res) if func.__name__ != 'quartile'\
                # else [float(res[0]), float(res[1])]

        return wrapper

    @DataChecker
    def mean(self, data):
        _size = len(data)
        return sum(data)/_size if _size else None

    @DataChecker
    def percentile(self, data, p):
        serie = sorted(data)
        if len(serie) == 0:
            return None
        k = (len(serie) - 1) * (p / 100)
        f = math.floor(k)
        c = math.ceil(k)
        if p > 50:
            return (serie[int(c)])
        elif p < 50:
            return (serie[int(f)])
        elif p == 50:
            return serie[int(k)]
    # @DataChecker
    # def percentile(self, data, p):
    #     serie = sorted(data)
    #     if len(serie) == 0:
    #         return None
    #     k = (len(serie) - 1) * (p / 100)
    #     f = math.floor(k)
    #     c = math.ceil(k)
    #     if f == c:
    #         return serie[int(k)]
    #     d0 = serie[int(f)] * (c - k)
    #     d1 = serie[int(c)] * (k - f)
    #     return (d0 + d1)

    @DataChecker
    def median(self, data):
        serie = sorted(data)
        f = math.floor((len(serie) - 1) * (50 / 100))
        c = math.ceil((len(serie) - 1) * (50 / 100))
        if f == c:
            return serie[int((len(serie) - 1) * (50 / 100))]
        return (serie[int(f)] * (c - (len(serie) - 1) * (50 / 100))
                + serie[int(c)] * ((len(serie) - 1) * (50 / 100) - f))
    # @DataChecker
    # def median(self, data):
    #     return self.percentile(data, 50) if len(data) else None

    @DataChecker
    def quartile(self, data):
        serie = sorted(data)
        return [serie[math.floor((len(serie) - 1) * (25 / 100))],
                serie[math.ceil((len(serie) - 1) * (75 / 100))]]\
                    if len(data) else None
    # @DataChecker
    # def quartile(self, data):
    #     return [self.percentile(data, 25), self.percentile(data, 75)] if len(data) else None

    @DataChecker
    def _sum(self, data):
        mean = self.mean(data)
        return sum(map(lambda x: (x - mean)**2, data))

    @DataChecker
    def var(self, data):
        return round(self._sum(data)/(len(data) - 1))\
            if len(data) and len(data) != 1 else None
    # @DataChecker
    # def var(self, data):
    #     return self._sum(data)/(len(data) - 1) if len(data) and len(data) != 1 else None

    @DataChecker
    def std(self, data):
        return math.sqrt(self._sum(data)/(len(data) - 1))\
            if len(data) and len(data) != 1 else None
    # @DataChecker
    # def std(self, data):
        # return math.sqrt(self._sum(data)/(len(data) - 1)) if len(data) and len(data) != 1 else None


# if __name__ == '__main__':
#     x = float("nan")
#     a = [1, 42, 300, 10, 59]
#     print(TinyStatistician().mean(a))
#     # Output:
#     # 82.4
#     print(TinyStatistician().median(a))
#     # Output:
#     # 42.0
#     print(TinyStatistician().quartile(a))
#     # Output:
#     # [10.0, 59.0]
#     print(TinyStatistician().percentile(a, 10))
#     # Output:
#     # 4.6
#     print(TinyStatistician().percentile(a, 15))
#     # Output:
#     # 6.4
#     print(TinyStatistician().percentile(a, 20))
#     # Output:
#     # 8.2
#     print(TinyStatistician().var(a))
#     # Output:
#     # 15349.3
#     print(TinyStatistician().std(a))
#     # Output:
#     # 123.89229193133849

# if __name__ == '__main__':

#     import TinyStatistician as ts
#     import numpy as np
#     data = [42, 7, 69, 18, 352, 3, 650, 754, 438, 2659]
#     epsilon = 1e-5
#     err = "Error, grade 0 :("
#     tstat = ts.TinyStatistician()
#     assert abs(tstat.mean(data) - 499.2) < epsilon, err
#     assert abs(tstat.median(data) - 210.5) < epsilon, err
#     quartile = tstat.quartile(data)
#     assert abs(quartile[0] - 18) < epsilon, err
#     assert abs(quartile[1] - 650) < epsilon, err
#     assert abs(tstat.percentile(data, 10) - 3) < epsilon, err
#     assert abs(tstat.percentile(data, 28) - 18) < epsilon, err
#     assert abs(tstat.percentile(data, 83) - 754) < epsilon, err
#     assert abs(tstat.var(data) - 654661) < epsilon, err
#     assert abs(tstat.std(data) - 809.11) < epsilon, err
