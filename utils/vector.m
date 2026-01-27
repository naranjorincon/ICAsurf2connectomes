function L = vector(m)     %better variable names and function name advised
   L = m(triu(true(size(m)), 1));
end