import numpy as np

# new version of polyfit from Greg
def polyfit2(times, data, deg = 1, errors = None, taylor=False,t0=None):
    '''
    This fits for the coefficents of the form: p[0]+ p[1]*x + p[2]*x**2 ...

    Keywords
    --------
    deg    - degree of the polynomial (Default: 1)
    taylor - include the coeffients from the Taylor series in the fit
             (1/2*p[0]x**2 + 1/6*p[1]*x**3, etc...). This is useful things like
             acceleration fits (default: False).

    Returns
    -------
    A tuple with (coeffients, covariance matrix)

    The coefficients are [p[0],p[1],p[2].... ]
    NOTE: This is in reverse order from numpy.polyfit. Remember to reverse the order
    for functions like np.polyval (np.polyval(times,np.flip(fit[0],0)))

    History
    -------
    2017-03-13 - Created by G. Martinez
    2017-03-14 - added 'taylor' keyword - T. Do
    '''

    N = int(deg)
    cov = np.zeros((N+1, N+1))
    a = np.zeros(N+1)
    if t0 is None:
        t0 = np.average(times, weights=[1.0/e/e for e in errors])
    faci = 1

    for i in range(N+1):
        facj = 1
        a[i] = np.sum([(d*(t - t0)**i)/e/e/faci for d, t, e in zip(data, times, errors)])
        cov[i,i] = np.sum([((t-t0)**(2*i))/faci/faci/e/e for t, e in zip(times, errors)])
        for j in range(i):
            cov[i,j] = np.sum([((t-t0)**(i+j))/faci/facj/e/e for t, e in zip(times, errors)])
            cov[j,i] = cov[i,j]
            if taylor:
                facj = facj*(j+1)
        if taylor:
            faci = faci*(i+1)

    cov = np.linalg.inv(cov)
    a = cov.dot(a)
    return (a, cov)
