mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[logger]\n\
level = 'debug'\n\
" > ~/.streamlit/config.toml
