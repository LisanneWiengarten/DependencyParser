from collections import deque

# Class for representing configurations
# IN: Stack and Buffer
class Configuration:
	def __init__(self, st, buff):
		# Stack o: last-in first-out append_back pop_front only
		self.stack = deque(st)
		# Buffer b: first-in first-out append_front pop_front only
		self.buffer = deque(buff)
		
	# Prettily writes configuration to string
	def write(self):
		out = "Stack: "
		for s in self.stack:
			out += s.write()
		out += "\n Buffer: "
		for b in self.buffer:
			out += b.write()
		return out