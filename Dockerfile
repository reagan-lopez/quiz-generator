FROM pgvector/pgvector:pg16

# Set environment variables
ENV POSTGRES_DB=prescreendb
ENV POSTGRES_USER=admin
ENV POSTGRES_PASSWORD=adminpassword

# Copy SQL script to initialize database
COPY init.sql /docker-entrypoint-initdb.d/

# Create a local volume to persist the data
VOLUME ["/var/lib/postgresql/data"]

# Expose the PostgreSQL default port
EXPOSE 5432
