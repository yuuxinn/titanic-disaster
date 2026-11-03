# R container for the Titanic logistic regression model
FROM rocker/r-ver:4.4.1

WORKDIR /app

# Install packages first so Docker can cache this layer
COPY install_packages.R .
RUN Rscript install_packages.R

# Copy the script (the data is NOT copied - it is mounted at run time)
COPY main.R .

CMD ["Rscript", "main.R"]
