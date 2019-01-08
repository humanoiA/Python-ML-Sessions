ex=[0.0,0.5,0.0,0.5,0.0]
pr=[0.2,0.3,0.4,0.1,0.6]
forcast_error=[ex[i]-pr[i]for i in range(len(ex))]
print(forcast_error)