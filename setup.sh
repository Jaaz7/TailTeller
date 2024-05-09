mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[logger]\n\
level = 'info'\n\
" > ~/.streamlit/config.toml
