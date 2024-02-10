-- Create UserInfo table
CREATE TABLE IF NOT EXISTS user_info (
    id BIGSERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    hashed_password TEXT NOT NULL,
    user_type TEXT NOT NULL
);

-- Create JobInfo table
CREATE TABLE IF NOT EXISTS job_info (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES user_info(id) ON DELETE CASCADE,
    job_role TEXT,
    job_description TEXT
);

CREATE EXTENSION vector;
-- Create JobCorpus table
CREATE TABLE IF NOT EXISTS job_corpus (
    id BIGSERIAL PRIMARY KEY,
    job_id BIGINT REFERENCES job_info(id) ON DELETE CASCADE,
    corpus_metadata TEXT,
    corpus_embedding vector(768)
);

-- Create Question table
CREATE TABLE IF NOT EXISTS question (
    id BIGSERIAL PRIMARY KEY,
    job_id BIGINT REFERENCES job_info(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    options TEXT[] NOT NULL,
    answers BOOLEAN[] NOT NULL
);

-- Create Response table
CREATE TABLE IF NOT EXISTS response (
    id BIGSERIAL PRIMARY KEY,    
    user_id BIGINT REFERENCES user_info(id) ON DELETE CASCADE,
    question_id BIGINT REFERENCES question(id) ON DELETE CASCADE,
    answers BOOLEAN[] NOT NULL
);

-- Create Result table
CREATE TABLE IF NOT EXISTS result (
    id BIGSERIAL PRIMARY KEY,
    job_id BIGINT REFERENCES job_info(id) ON DELETE CASCADE,    
    question_id BIGINT REFERENCES question(id) ON DELETE CASCADE,
    result DECIMAL NOT NULL
);